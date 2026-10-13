// Separate arithmetic oracle: it does not read candidate DOM or application code.
export const observations=Object.freeze([
  ["A1","A",2],["A2","A",4],["A3","A",8],
  ["B1","B",1],["B2","B",3],["B3","B",9]
]);
export function expected(group,minimum,maximum){
  const rows=observations.filter(([,g,value])=>(group==="all"||g===group)&&value>=minimum&&value<=maximum);
  return {ids:rows.map(([id])=>id),count:rows.length,mean:rows.length?rows.reduce((total,row)=>total+row[2],0)/rows.length:null,values:rows.map(([,g,value])=>value)};
}
export function parseCsv(text){
  const lines=text.trimEnd().split(/\r?\n/).map(line=>line.split(","));
  if(lines[0]?.join(",")!=="row_id,group,measurement")throw new Error("Unexpected CSV header");
  return lines.slice(1).map(([id,group,value])=>[id,group,Number(value)]);
}
