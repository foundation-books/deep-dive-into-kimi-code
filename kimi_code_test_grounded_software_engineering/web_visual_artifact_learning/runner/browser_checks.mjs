// Dependency-free Chrome DevTools Protocol runner for the controlled dashboard.
// The evaluator stays outside app/ and the versioned candidate files.
import fs from "node:fs/promises";
import {constants,existsSync} from "node:fs";
import os from "node:os";
import path from "node:path";
import {fileURLToPath,pathToFileURL} from "node:url";
import {spawn,spawnSync} from "node:child_process";
import {expected,parseCsv} from "../oracle.mjs";

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),"..");
const output=path.resolve(process.argv[2]??path.join(root,"evidence"));
const expectedChromeBuild="153.0.8010.53";
const viewport={desktop:[960,900],narrow:[390,844],narrowFinal:[430,932]};
const capturedScreenshots=[];
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const hash=async file=>{const {createHash}=await import("node:crypto");return createHash("sha256").update(await fs.readFile(file)).digest("hex")};
function insist(value,message){if(!value)throw new Error(message)}

async function isExecutable(file){
  try{
    if(!(await fs.stat(file)).isFile())return false;
    await fs.access(file,constants.X_OK);
    return true;
  }catch{return false}
}

// Resolve the reader's executable without embedding installation directories.
export async function resolveChrome(env=process.env){
  const override=env.K3_DASHBOARD_CHROME?.trim();
  const names=override?[override]:["google-chrome","google-chrome-stable","chromium","chromium-browser","chrome","chrome.exe","chromium.exe"];
  const directories=(env.PATH??env.Path??"").split(path.delimiter).filter(Boolean);
  for(const name of names){
    const files=path.isAbsolute(name)||name.includes("/")||name.includes("\\")
      ?[path.resolve(name)]:directories.map(directory=>path.resolve(directory,name));
    for(const file of files)if(await isExecutable(file))return file;
  }
  throw new Error(override
    ?"K3_DASHBOARD_CHROME must identify an executable file or a command on PATH."
    :"No Chrome or Chromium executable found. Add it to PATH or set K3_DASHBOARD_CHROME to its executable path.");
}

export function checkChromeVersion(product){
  const matches=["Chrome","HeadlessChrome","Chromium"].some(name=>product===`${name}/${expectedChromeBuild}`);
  insist(matches,`Expected Chrome or Chromium build ${expectedChromeBuild}; found ${product}. Re-capture and review the reference before changing the pin.`);
  return product;
}

async function stopBrowser(cdp,proc){
  cdp?.close();
  if(proc.exitCode!==null||proc.signalCode!==null)return;
  const exited=new Promise(resolve=>proc.once("exit",resolve));
  proc.kill();
  await Promise.race([exited,pause(1000)]);
}

class CDP {
  constructor(ws){this.ws=ws;this.id=0;this.pending=new Map();this.exceptions=[];
    ws.addEventListener("message",event=>{const data=JSON.parse(event.data);
      if(data.method==="Runtime.exceptionThrown")this.exceptions.push(data.params.exceptionDetails.text);
      if(data.id&&this.pending.has(data.id)){const {resolve,reject}=this.pending.get(data.id);this.pending.delete(data.id);data.error?reject(new Error(data.error.message)):resolve(data.result)}
    });
  }
  static async connect(url){const ws=new WebSocket(url);await new Promise((resolve,reject)=>{ws.addEventListener("open",resolve,{once:true});ws.addEventListener("error",reject,{once:true})});return new CDP(ws)}
  call(method,params={}){const id=++this.id;return new Promise((resolve,reject)=>{this.pending.set(id,{resolve,reject});this.ws.send(JSON.stringify({id,method,params}))})}
  async eval(expression){const result=await this.call("Runtime.evaluate",{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw new Error(result.exceptionDetails.text);return result.result.value}
  close(){this.ws.close()}
}

async function launchBrowser(temp){
  const chrome=await resolveChrome();
  const profile=path.join(temp,"chrome-profile");await fs.mkdir(profile);
  const proc=spawn(chrome,["--headless=new","--no-first-run","--disable-default-apps","--disable-background-networking","--disable-extensions","--disable-gpu","--hide-scrollbars","--remote-debugging-port=0",`--user-data-dir=${profile}`,"about:blank"],{stdio:"ignore"});
  let cdp,launchError;
  proc.once("error",error=>{launchError=error});
  try{
    const portFile=path.join(profile,"DevToolsActivePort");
    for(let i=0;i<200&&!existsSync(portFile);i++){
      if(launchError)throw launchError;
      if(proc.exitCode!==null||proc.signalCode!==null)throw new Error(`Chrome exited ${proc.signalCode??proc.exitCode}`);
      await pause(50);
    }
    insist(existsSync(portFile),"Chrome DevTools port unavailable");
    const port=Number((await fs.readFile(portFile,"utf8")).split("\n")[0]);
    const tabs=await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    const page=tabs.find(tab=>tab.type==="page");insist(page,"Chrome page unavailable");
    cdp=await CDP.connect(page.webSocketDebuggerUrl);
    const version=checkChromeVersion((await cdp.call("Browser.getVersion")).product);
    await cdp.call("Page.enable");await cdp.call("Runtime.enable");
    await cdp.call("Emulation.setTimezoneOverride",{timezoneId:"UTC"});
    await cdp.call("Emulation.setLocaleOverride",{locale:"en-US"});
    await cdp.call("Page.setDownloadBehavior",{behavior:"allow",downloadPath:path.join(temp,"downloads")});
    return {cdp,proc,version,temp};
  }catch(error){await stopBrowser(cdp,proc);throw error}
}
async function materialize(name,temp){
  const dest=path.join(temp,name);await fs.mkdir(dest,{recursive:true});
  for(const file of ["index.html","style.css"]){await fs.copyFile(path.join(root,"app",file),path.join(dest,file))}
  if(name==="reference"){
    await fs.writeFile(path.join(dest,"candidate.js"),"// Independent reference application.\n");
    await fs.copyFile(path.join(root,"reference","reference.js"),path.join(dest,"app.js"));
  }else{
    await fs.copyFile(path.join(root,"variants",`${name}.js`),path.join(dest,"candidate.js"));
    await fs.copyFile(path.join(root,"app","app.js"),path.join(dest,"app.js"));
  }
  return dest;
}
async function view(browser,dir,size){
  const {cdp}=browser;await cdp.call("Emulation.setDeviceMetricsOverride",{width:size[0],height:size[1],deviceScaleFactor:1,mobile:false});
  cdp.exceptions=[];await cdp.call("Page.navigate",{url:pathToFileURL(path.join(dir,"index.html")).href});
  let ready=false;for(let i=0;i<100;i++){ready=await cdp.eval("document.documentElement?.dataset.ready==='true'").catch(()=>false);if(ready)break;await pause(30)}
  return {ready,exceptions:[...cdp.exceptions]};
}
async function setFilters(cdp,group,min,max){
  await cdp.eval(`(()=>{const values=${JSON.stringify({group,min,max})};for(const [id,value] of Object.entries({group:values.group,minimum:values.min,maximum:values.max})){const el=document.getElementById(id);el.value=String(value);el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}))}return true})()`);
}
async function state(cdp){return cdp.eval(`(()=>({ids:[...document.querySelectorAll('#rows tr')].map(row=>row.cells[0].textContent),bars:[...document.querySelectorAll('.bar-id')].map(el=>el.textContent),count:document.getElementById('count').textContent,mean:document.getElementById('mean').textContent,emptyChart:!document.getElementById('empty-chart').hidden,emptyTable:!document.getElementById('empty-table').hidden,chartHidden:document.getElementById('bars').hidden,selection:document.getElementById('selection-label').textContent}))()`)}
const same=(left,right)=>JSON.stringify(left)===JSON.stringify(right);
function rowCorrect(actual,want){return same(actual.ids,want.ids)&&same(actual.bars,want.ids)}
function summaryCorrect(actual,want){return actual.count===String(want.count)&&actual.mean===(want.mean===null?"—":String(want.mean))}
async function screenshot(cdp,file){const {data}=await cdp.call("Page.captureScreenshot",{format:"png",captureBeyondViewport:false});await fs.writeFile(file,Buffer.from(data,"base64"));capturedScreenshots.push({file:path.basename(file),sha256:await hash(file)});return file}
async function visual(cdp,narrow){
  return cdp.eval(`(()=>{
    const ids=['group','minimum','maximum','export','count','mean','bars','rows'];
    const rect=id=>document.getElementById(id).getBoundingClientRect();
    const usable=id=>{const el=document.getElementById(id),r=el.getBoundingClientRect(),x=r.left+r.width/2,y=r.top+r.height/2,top=document.elementFromPoint(x,y);return r.width>0&&r.height>0&&x>=0&&x<innerWidth&&y>=0&&y<innerHeight&&(top===el||el.contains(top))};
    const visible=id=>{const el=document.getElementById(id),r=rect(id),s=getComputedStyle(el);return r.width>0&&r.height>0&&r.left>=0&&r.right<=innerWidth+1&&s.visibility==='visible'&&s.display!=='none'};
    const columns=getComputedStyle(document.querySelector('.workspace')).gridTemplateColumns.split(' ').length;
    return {required:['group','minimum','maximum','export','count','mean','legend','rows'].every(visible),legend:visible('legend'),controls:['group','minimum','maximum','export'].every(usable),results:['count','mean','bars','rows'].every(visible),noOverflow:document.documentElement.scrollWidth<=innerWidth+1,columns,readable:['count','mean','table-title'].every(id=>parseFloat(getComputedStyle(document.getElementById(id)).fontSize)>=12),font:getComputedStyle(document.body).fontFamily,fontsReady:document.fonts.status==='loaded'}
  })()`);
}
async function clickFocus(cdp,id){
  const center=await cdp.eval(`(()=>{const r=document.getElementById(${JSON.stringify(id)}).getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2}})()`);
  await cdp.call("Input.dispatchMouseEvent",{type:"mousePressed",x:center.x,y:center.y,button:"left",clickCount:1});
  await cdp.call("Input.dispatchMouseEvent",{type:"mouseReleased",x:center.x,y:center.y,button:"left",clickCount:1});
  const focused=await cdp.eval(`document.activeElement?.id===${JSON.stringify(id)}`);
  await cdp.call("Input.dispatchKeyEvent",{type:"keyDown",key:"Escape",code:"Escape",windowsVirtualKeyCode:27});
  await cdp.call("Input.dispatchKeyEvent",{type:"keyUp",key:"Escape",code:"Escape",windowsVirtualKeyCode:27});
  return focused;
}
async function csvDownload(browser,group,min,max){
  const {cdp,temp}=browser;const downloads=path.join(temp,"downloads");await fs.mkdir(downloads,{recursive:true});
  const file=path.join(downloads,"selected-measurements.csv");await fs.rm(file,{force:true});
  await setFilters(cdp,group,min,max);await cdp.eval("document.getElementById('export').click();true");
  for(let i=0;i<100&&!existsSync(file);i++)await pause(30);
  insist(existsSync(file),"CSV download not produced");
  const content=await fs.readFile(file,"utf8");return {rows:parseCsv(content),content};
}
async function runCandidate(browser,name,dir){
  const {cdp}=browser,screenDir=path.join(output,"screenshots");
  const initial=await view(browser,dir,viewport.desktop);
  const W1=initial.ready&&initial.exceptions.length===0;
  if(!W1)return {name,W1,error:initial};
  const defaultVisual=await visual(cdp,false);
  if(name==="reference"||name==="D0"||name==="D2"||name==="D3"||name==="D4")await screenshot(cdp,path.join(screenDir,`${name}-desktop-default.png`));
  defaultVisual.pointerFocus=await clickFocus(cdp,"group");
  await setFilters(cdp,"A",3,8);const a=await state(cdp),aWant=expected("A",3,8);
  if(name==="D0"||name==="D3")await screenshot(cdp,path.join(screenDir,`${name}-A-3-8.png`));
  await setFilters(cdp,"B",3,8);const b=await state(cdp),bWant=expected("B",3,8);
  await setFilters(cdp,"A",2,4);const boundary=await state(cdp),boundaryWant=expected("A",2,4);
  const exportA=await csvDownload(browser,"A",3,8);
  const exportB=await csvDownload(browser,"B",3,8);
  await setFilters(cdp,"A",9,10);const empty=await state(cdp);
  const W2=same(a.ids,aWant.ids)&&same(b.ids,bWant.ids);
  const W3=same(b.ids,["B2"])&&same(boundary.ids,boundaryWant.ids)&&same(a.ids,["A2","A3"]);
  const W4data=rowCorrect(a,aWant)&&rowCorrect(b,bWant)&&rowCorrect(boundary,boundaryWant);
  const W4summary=summaryCorrect(a,aWant)&&summaryCorrect(b,bWant)&&summaryCorrect(boundary,boundaryWant);
  const W5=same(exportA.rows.map(row=>row[0]),aWant.ids)&&same(exportB.rows.map(row=>row[0]),bWant.ids);
  const W6=empty.count==="0"&&empty.mean==="—"&&empty.ids.length===0&&empty.bars.length===0&&empty.emptyChart&&empty.emptyTable&&empty.chartHidden;
  const W7=defaultVisual.required&&defaultVisual.controls&&defaultVisual.pointerFocus&&defaultVisual.results&&defaultVisual.noOverflow&&defaultVisual.columns===2&&defaultVisual.readable;
  const narrowChecks=[];
  // The 430 x 932 case is reserved for the rebuilt selected source below.
  for(const key of ["narrow"]){
    const opened=await view(browser,dir,viewport[key]);
    insist(opened.ready&&opened.exceptions.length===0,`${name} failed to load at ${key}; fixture unresolved`);
    await setFilters(cdp,"A",3,8);
    const narrow=await visual(cdp,true);
    if((name==="D1"||name==="D3"||name==="D4")&&key==="narrow"){
      await screenshot(cdp,path.join(screenDir,`${name}-narrow-A-3-8.png`));
    }
    narrow.pointerFocus=await clickFocus(cdp,"group");
    narrowChecks.push(narrow);
  }
  const W8=narrowChecks.every(v=>v.required&&v.controls&&v.pointerFocus&&v.results&&v.noOverflow&&v.columns===1&&v.readable);
  const functional=[W2,W3,W4data,W4summary,W5,W6];
  const visualCriteria={elements:defaultVisual.required,legible:defaultVisual.readable&&narrowChecks.every(v=>v.readable&&v.results),layout:defaultVisual.columns===2&&narrowChecks.every(v=>v.columns===1&&v.noOverflow),usable:defaultVisual.controls&&defaultVisual.pointerFocus&&narrowChecks.every(v=>v.controls&&v.pointerFocus)};
  const f=functional.filter(Boolean).length/6,v=Object.values(visualCriteria).filter(Boolean).length/4;
  const score=.8*f+.2*v;
  const eligible=W1&&functional.every(Boolean)&&W7&&W8;
  return {name,W1,W2,W3,W4data,W4summary,W5,W6,W7,W8,f,v,score,eligible,visualCriteria,observed:{A:a,B:b,boundary,empty,exportA:exportA.rows,exportB:exportB.rows,desktop:defaultVisual,narrow:narrowChecks}};
}
async function finalEvaluation(browser,dir){
  const {cdp}=browser;let result=await view(browser,dir,viewport.desktop);
  insist(result.ready&&result.exceptions.length===0,"Selected source failed clean rebuild");
  await setFilters(cdp,"A",3,8);const first=await state(cdp);
  await setFilters(cdp,"B",1,3);const second=await state(cdp),want=expected("B",1,3);
  const file=await csvDownload(browser,"B",1,3);
  result=await view(browser,dir,viewport.narrowFinal);
  insist(result.ready&&result.exceptions.length===0,"Selected source failed final narrow rebuild");
  await setFilters(cdp,"B",1,3);
  const narrow=await visual(cdp,true),after=await state(cdp);narrow.pointerFocus=await clickFocus(cdp,"group");
  return {passed:rowCorrect(first,expected("A",3,8))&&rowCorrect(second,want)&&summaryCorrect(second,want)&&same(file.rows.map(row=>row[0]),want.ids)&&rowCorrect(after,want)&&narrow.required&&narrow.controls&&narrow.pointerFocus&&narrow.results&&narrow.noOverflow,first,second,export:file.rows,narrow};
}
async function positionProbe(browser,dir,temp){
  const {cdp}=browser;
  const measure=()=>cdp.eval(`(()=>{const r=document.getElementById('chart-title').getBoundingClientRect();return {x:(r.left+r.width/2)/innerWidth,y:(r.top+r.height/2)/innerHeight,width:r.width/innerWidth,height:r.height/innerHeight}})()`);
  let status=await view(browser,dir,viewport.desktop);
  insist(status.ready&&status.exceptions.length===0,"Position baseline failed to load");
  const baseline=await measure();
  await screenshot(cdp,path.join(output,"screenshots","metric-baseline.png"));
  const layoutDir=await materialize("D3",path.join(temp,"metric-layout"));
  await fs.appendFile(path.join(layoutDir,"style.css"),"\n#chart-title{transform:translateX(48px)}\n");
  status=await view(browser,layoutDir,viewport.desktop);
  insist(status.ready&&status.exceptions.length===0,"Position layout intervention failed to load");
  const layout=await measure();
  await screenshot(cdp,path.join(output,"screenshots","metric-layout.png"));
  status=await view(browser,dir,viewport.narrow);
  insist(status.ready&&status.exceptions.length===0,"Position capture intervention failed to load");
  const capture=await measure();
  await screenshot(cdp,path.join(output,"screenshots","metric-capture.png"));
  return {source:"D3",filter:"default",block:"chart-title",boxSource:"DOM geometry normalized by viewport, not Design2Code OCR",layoutIntervention:"append #chart-title{transform:translateX(48px)} to a disposable style copy",baseline:{viewport:viewport.desktop,box:baseline},layout:{viewport:viewport.desktop,box:layout},capture:{viewport:viewport.narrow,box:capture}};
}
async function main(){
  const syntax=[path.join(root,"app","app.js"),path.join(root,"reference","reference.js"),...Array.from({length:5},(_,i)=>path.join(root,"variants",`D${i}.js`))];
  for(const file of syntax){const check=spawnSync(process.execPath,["--check",file],{encoding:"utf8"});insist(check.status===0,`JavaScript syntax: ${file}: ${check.stderr}`)}
  await fs.mkdir(path.join(output,"screenshots"),{recursive:true});
  const temp=await fs.mkdtemp(path.join(os.tmpdir(),"k3-dashboard-"));
  let browser;
  try{
    browser=await launchBrowser(temp);
    const names=["reference","D0","D1","D2","D3","D4"];
    const dirs=Object.fromEntries(await Promise.all(names.map(async name=>[name,await materialize(name,temp)])));
    const runs=[];for(const name of names)runs.push(await runCandidate(browser,name,dirs[name]));
    insist(runs[0].eligible,"Independent reference failed declared checks");
    const candidates=runs.slice(1),eligible=candidates.filter(run=>run.eligible).sort((a,b)=>b.score-a.score||a.name.localeCompare(b.name));
    insist(eligible.length>0,"No candidate meets delivery requirements");
    const selected=eligible[0].name;
    // Rebuild the selected source after selection; final cases cannot choose it.
    const rebuilt=await materialize(selected,path.join(temp,"final"));
    const final=await finalEvaluation(browser,rebuilt);
    const metricControl=await positionProbe(browser,dirs.D3,temp);
    const sourceHashes={};for(const name of names){sourceHashes[name]={candidate:await hash(path.join(dirs[name],"candidate.js")),app:await hash(path.join(dirs[name],"app.js")),style:await hash(path.join(dirs[name],"style.css"))}}
    const report={evidenceClass:"Executed controlled teaching fixture; no K3 model episode",browser:browser.version,node:process.version,environment:{deviceScaleFactor:1,locale:"en-US",timezone:"UTC",viewport,hideScrollbars:true,font:"Arial, Helvetica, sans-serif",network:"disabled by local file fixture"},runnerSha256:await hash(fileURLToPath(import.meta.url)),sourceHashes,capturedScreenshots,runs,selection:{rule:"Among W1-W8 eligible candidates, highest 0.8f+0.2v; earliest D number breaks ties",selected,last:"D4",lastEligible:candidates.at(-1).eligible},final,metricControl};
    await fs.writeFile(path.join(output,"results.json"),JSON.stringify(report,null,2)+"\n");
    console.log(JSON.stringify({browser:browser.version,matrix:runs.map(({name,W1,W2,W3,W4data,W4summary,W5,W6,W7,W8,f,v,eligible})=>({name,W1,W2,W3,W4data,W4summary,W5,W6,W7,W8,f,v,eligible})),selection:report.selection,finalPassed:final.passed},null,2));
    insist(final.passed,"Selected source failed independent final cases");
  }finally{
    if(browser)await stopBrowser(browser.cdp,browser.proc);
    await fs.rm(temp,{recursive:true,force:true,maxRetries:10,retryDelay:100});
  }
}
if(process.argv[1]&&pathToFileURL(path.resolve(process.argv[1])).href===import.meta.url){
  main().catch(error=>{console.error(error.stack);process.exitCode=1});
}
