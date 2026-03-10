fetch('run_data.json').then(r=>r.json()).then(d=>{document.getElementById('out').textContent=JSON.stringify(d.summary,null,2);});
