
/* ──────────────────────────────────────
   PAGE UTILITIES
────────────────────────────────────── */
document.querySelectorAll('a[href="#subjects"]').forEach(el=>{
  el.addEventListener('click',e=>{e.preventDefault();document.getElementById('subjects').scrollIntoView({behavior:'smooth',block:'start'})});
});
document.addEventListener('keydown',e=>{
  if((e.metaKey||e.ctrlKey)&&e.key==='k'){e.preventDefault();document.querySelector('.srch input').focus()}
});
const ioObs=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){e.target.style.animationPlayState='running';ioObs.unobserve(e.target)}})},{threshold:.1});
document.querySelectorAll('.sc,.tc-card,.nr').forEach(el=>{el.style.animationPlayState='paused';ioObs.observe(el)});

/* ──────────────────────────────────────
   CHATBOT STATE
────────────────────────────────────── */
// The NVIDIA API key lives server-side in chat-proxy.php — never put it here.
// This just points at your own proxy file, uploaded next to chatbot.js on InfinityFree.
const CHAT_ENDPOINT = 'chat-proxy.php';
const NVIDIA_MODEL  = 'openai/gpt-oss-120b';
// Other free models to try: 'meta/llama-4-maverick-17b-128e-instruct' (friendly, multilingual),
//                            'qwen/qwen2.5-7b-instruct' (fast, light fallback)

let cbOpen      = false;
let cbBusy      = false;
let cbSubj      = 'All Subjects';
let cbHistory   = [];  // {role,content}[]

const SUGGS = {
  'All Subjects':    ['Explain lens formula derivation','SN1 vs SN2 difference','Integration by parts formula','Mitosis vs Meiosis'],
  'Physics':         ['Derive mirror formula step by step','Explain Gauss\'s Law with example','What is photoelectric effect?','Biot-Savart Law vs Ampere\'s Law'],
  'Chemistry':       ['SN1 vs SN2 mechanism difference','Explain electrochemical series','What is Aldol condensation?','Raoult\'s law with example'],
  'Mathematics':     ['Integration by parts with example','Explain continuity and differentiability','How to find eigen values?','3D geometry distance formula'],
  'Biology':         ['Mitosis vs Meiosis comparison','Steps of DNA replication','Explain Krebs cycle','What is ecosystem energy flow?'],
  'English':         ['Summary of The Last Lesson','Formal letter writing format','How to write a notice?','Explain Deep Water theme'],
  'Computer Science':['Python list vs tuple difference','Explain SQL JOIN types','What is OOP in Python?','Recursion with example code'],
  'Economics':       ['Law of demand with diagram','Difference GDP vs GNP','Perfect competition features','Types of money supply M1 M2'],
  'Accountancy':     ['Partnership deed format','Goodwill valuation methods','Cash flow statement steps','Share forfeiture journal entry'],
};

/* ──────────────────────────────────────
   TOGGLE
────────────────────────────────────── */
function cbToggle(){
  cbOpen=!cbOpen;
  document.getElementById('cb-win').classList.toggle('open',cbOpen);
  const fab=document.getElementById('cb-fab');
  fab.style.transform = cbOpen ? 'scale(.9)' : '';
  if(cbOpen) setTimeout(()=>document.getElementById('cb-input').focus(),300);
  document.exitFullscreen()
}


/* ──────────────────────────────────────
   SUBJECT FILTER
────────────────────────────────────── */
function cbSubject(el, subj){
  document.querySelectorAll('#cb-chips .cb-chip').forEach(c=>c.classList.remove('on'));
  el.classList.add('on');
  cbSubj = subj;
  const s = SUGGS[subj]||SUGGS['All Subjects'];
  document.getElementById('cb-suggest').innerHTML = s.map(t=>`<span class="sug-btn" onclick="cbUseSug(this)">${t}</span>`).join('');
}

function cbUseSug(el){
  document.getElementById('cb-input').value = el.textContent;
  cbSend();
}

/* ──────────────────────────────────────
   HELPERS
────────────────────────────────────── */
function cbResize(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,110)+'px'}
function cbKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();cbSend()}}
function cbTime(){return new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}

function cbFmt(t){
  // escape HTML then apply markdown-lite
  return t
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>')
    .replace(/\*(.*?)\*/g,'<em>$1</em>')
    .replace(/`([^`]+)`/g,'<code>$1</code>')
    .replace(/\n/g,'<br>');
}

function cbAppend(role, text){
  const msgs = document.getElementById('cb-msgs');
  const d = document.createElement('div');
  d.className = `msg ${role==='user'?'me':'bot'}`;
  const av = role==='user'?'👤':'🤖';
  d.innerHTML=`<div class="msg-av-sm">${av}</div><div class="msg-wrap"><div class="msg-bbl">${cbFmt(text)}</div><div class="msg-ts">${cbTime()}</div></div>`;
  msgs.appendChild(d);
  msgs.scrollTop=msgs.scrollHeight;
}

function cbShowTyping(){
  const msgs=document.getElementById('cb-msgs');
  const d=document.createElement('div');
  d.className='msg bot';d.id='cb-typing';
  d.innerHTML=`<div class="msg-av-sm">🤖</div><div class="msg-wrap"><div class="typing-ind"><span></span><span></span><span></span></div></div>`;
  msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight;
}
function cbHideTyping(){const t=document.getElementById('cb-typing');if(t)t.remove()}

function cbClear(){
  cbHistory=[];
  document.getElementById('cb-msgs').innerHTML='';
  cbAppend('bot','Chat cleared! 🧹 Ask me any Class 12 doubt and I\'ll help you out.');
}

/* ──────────────────────────────────────
   SEND MESSAGE  ←  CORE FUNCTION
────────────────────────────────────── */
async function cbSend(){
  const inp = document.getElementById('cb-input');
  const txt = inp.value.trim();
  if(!txt||cbBusy) return;

  inp.value=''; inp.style.height='auto';
  cbAppend('user', txt);

  // Push to history (keep last 16 turns)
  cbHistory.push({role:'user', content:txt});
  if(cbHistory.length>16) cbHistory=cbHistory.slice(-16);

  cbBusy=true;
  document.getElementById('cb-send').disabled=true;
  cbShowTyping();

  const subjLine = cbSubj!=='All Subjects' ? `The student is currently studying **${cbSubj}**.` : '';

  const SYS = `You are StudyBot, an expert and friendly Class 12 tutor for CBSE/ISC board exams (India).
${subjLine}
Guidelines:
- Answer Class 12 doubts accurately and clearly
- Subjects: Physics, Chemistry, Mathematics, Biology, English, Computer Science, Economics, Accountancy
- For numericals: show step-by-step solutions with formulas
- Use **bold** for key terms, *italics* for emphasis, and \`backticks\` for formulas or code snippets
- Keep answers concise but complete — 2 to 4 paragraphs max unless a full derivation is needed
- Always end with a short exam tip or memory trick when relevant
- Be warm, encouraging and motivating — students are preparing for board exams
- If a question is unclear, ask for clarification politely`;

  try {
    const resp = await fetch(CHAT_ENDPOINT,{
      method:'POST',
      credentials: 'same-origin', 
      headers:{ 'Content-Type':'application/json' },
      body: JSON.stringify({
        model: NVIDIA_MODEL,
        max_tokens:1000,
        messages:[
          {role:'system', content:SYS},
          ...cbHistory.map(m=>({role:m.role,content:m.content}))
        ]
      })
    });

    if(!resp.ok){
      const err=await resp.json().catch(()=>({}));
      throw new Error(err.error?.message||`HTTP ${resp.status}`);
    }

    const data = await resp.json();
    const reply = data.choices?.[0]?.message?.content?.trim()
      || "I couldn't generate a response. Please try again!";

    cbHideTyping();
    cbAppend('bot', reply);

    // Save assistant reply to history
    cbHistory.push({role:'assistant', content:reply});

  } catch(err){
    cbHideTyping();
    cbAppend('bot',`⚠️ **Connection issue.** Could not reach the AI right now.\n\n*${err.message}*\n\nPlease try again in a moment!`);
    console.error('StudyBot error:',err);
  }

  cbBusy=false;
  document.getElementById('cb-send').disabled=false;
  document.getElementById('cb-input').focus();
}
