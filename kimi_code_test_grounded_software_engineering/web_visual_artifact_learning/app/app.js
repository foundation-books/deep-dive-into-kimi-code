// Original, invented teaching measurements; no third-party data or assets.
const measurements = Object.freeze([
  {id:"A1",group:"A",value:2},{id:"A2",group:"A",value:4},
  {id:"A3",group:"A",value:8},{id:"B1",group:"B",value:1},
  {id:"B2",group:"B",value:3},{id:"B3",group:"B",value:9}
]);
const variant = window.dashboardVariant;
const $ = id => document.getElementById(id);
function requestedRows(){
  const group=$("group").value, lo=Number($("minimum").value), hi=Number($("maximum").value);
  return measurements.filter(row=>(group==="all"||row.group===group)&&row.value>=lo&&row.value<=hi);
}
function render(){
  const requested=requestedRows();
  // D0's constructed defect: labels change, while the data view stays stale.
  const rows=variant.filterMode==="labels-only"?measurements:requested;
  $("count").textContent=String(rows.length);
  $("mean").textContent=rows.length?String(rows.reduce((sum,row)=>sum+row.value,0)/rows.length):"—";
  $("selection-label").textContent=`${$("group").value==="all"?"All groups":"Group "+$("group").value} · ${$("minimum").value}–${$("maximum").value}`;
  $("rows").replaceChildren(...rows.map(row=>{
    const tr=document.createElement("tr");
    for(const content of [row.id,row.group,String(row.value)]){
      const td=document.createElement("td");td.textContent=content;tr.append(td);
    }
    tr.lastChild.className="numeric";return tr;
  }));
  $("bars").replaceChildren(...rows.map(row=>{
    const item=document.createElement("div");item.className="bar-wrap";
    item.innerHTML=`<span class="bar-value">${row.value}</span><span class="bar" style="height:${row.value*12}px"></span><span class="bar-id">${row.id}</span>`;
    return item;
  }));
  $("bars").hidden=rows.length===0;
  $("empty-chart").hidden=rows.length!==0;
  $("empty-table").hidden=rows.length!==0;
  $("bars").setAttribute("aria-label",rows.length?`Selected measurements: ${rows.map(row=>`${row.id} ${row.value}`).join(", ")}`:"No selected measurements");
}
function exportCsv(){
  // D0 and D1 deliberately export the original array, exposing stale state.
  const rows=variant.exportMode==="all"?measurements:requestedRows();
  const csv=["row_id,group,measurement",...rows.map(row=>`${row.id},${row.group},${row.value}`)].join("\r\n")+"\r\n";
  const href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
  const a=document.createElement("a");a.href=href;a.download="selected-measurements.csv";document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(href),1000);
}
for(const id of ["group","minimum","maximum"]){$(id).addEventListener("input",render);$(id).addEventListener("change",render)}
$("export").addEventListener("click",exportCsv);
if(variant.clipNarrow) document.documentElement.classList.add("clip-narrow");
if(variant.hideLegend) document.documentElement.classList.add("hide-legend");
if(variant.blockControls){const blocker=document.createElement("div");blocker.id="control-blocker";blocker.setAttribute("aria-hidden","true");$("group").parentElement.append(blocker);document.documentElement.classList.add("block-controls")}
render();document.documentElement.dataset.ready="true";
