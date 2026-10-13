// Independently written reference behavior for six original toy observations.
// The reference source is never copied into a candidate workspace.
const data=[
  ["A1","A",2],["A2","A",4],["A3","A",8],
  ["B1","B",1],["B2","B",3],["B3","B",9]
];
const element=id=>document.getElementById(id);
function selected(){
  const group=element("group").value;
  const min=Number(element("minimum").value);
  const max=Number(element("maximum").value);
  return data.filter(([id,g,value])=>(group==="all"||g===group)&&min<=value&&value<=max);
}
function draw(){
  const rows=selected();
  element("count").textContent=rows.length;
  element("mean").textContent=rows.length?rows.reduce((sum,row)=>sum+row[2],0)/rows.length:"—";
  element("selection-label").textContent=`${element("group").value==="all"?"All groups":"Group "+element("group").value} · ${element("minimum").value}–${element("maximum").value}`;
  element("rows").innerHTML=rows.map(([id,group,value])=>`<tr><td>${id}</td><td>${group}</td><td class="numeric">${value}</td></tr>`).join("");
  element("bars").innerHTML=rows.map(([id,group,value])=>`<div class="bar-wrap"><span class="bar-value">${value}</span><span class="bar" style="height:${value*12}px"></span><span class="bar-id">${id}</span></div>`).join("");
  element("bars").hidden=rows.length===0;
  element("empty-chart").hidden=rows.length!==0;
  element("empty-table").hidden=rows.length!==0;
  element("bars").setAttribute("aria-label",rows.length?`Selected measurements: ${rows.map(([id,g,value])=>`${id} ${value}`).join(", ")}`:"No selected measurements");
}
for(const id of ["group","minimum","maximum"]){element(id).addEventListener("input",draw);element(id).addEventListener("change",draw)}
element("export").addEventListener("click",()=>{
  const csv=["row_id,group,measurement",...selected().map(row=>row.join(","))].join("\r\n")+"\r\n";
  const url=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
  const link=document.createElement("a");link.href=url;link.download="selected-measurements.csv";document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
draw();document.documentElement.dataset.ready="true";
