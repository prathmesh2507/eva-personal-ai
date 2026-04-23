import { useState, useRef, useEffect } from "react";


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CONSTANTS & CONFIG
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const MODEL = "claude-sonnet-4-20250514";

const INTENT_MAP = {
  SYSTEM:        { color: "#ff6b35", icon: "⚙️",  label: "SYSTEM"   },
  CODING:        { color: "#a855f7", icon: "💻",  label: "CODING"   },
  EMOTIONAL:     { color: "#ec4899", icon: "💙",  label: "EMOTIONAL"},
  MOTIVATION:    { color: "#f59e0b", icon: "🔥",  label: "HYPE"     },
  COMMUNICATION: { color: "#10b981", icon: "📨",  label: "COMM"     },
  PERSONALITY:   { color: "#06b6d4", icon: "😏",  label: "VIBE"     },
  GENERAL:       { color: "#6b7280", icon: "💬",  label: "CHAT"     },
};

const MOOD_MAP = {
  happy:   { emoji: "😊", color: "#10b981", label: "Happy"   },
  neutral: { emoji: "😐", color: "#6b7280", label: "Neutral" },
  sad:     { emoji: "😢", color: "#3b82f6", label: "Low"     },
  stressed:{ emoji: "😰", color: "#ef4444", label: "Stressed"},
  excited: { emoji: "🔥", color: "#f59e0b", label: "Excited" },
  focused: { emoji: "🎯", color: "#a855f7", label: "Focused" },
};

const SYS_CMDS = [
  { triggers:["open chrome","launch chrome","chrome"],         icon:"🌐", name:"Google Chrome",    msg:"Launching browser…"         },
  { triggers:["open vscode","open vs code","vscode"],          icon:"💻", name:"VS Code",           msg:"Launching editor…"          },
  { triggers:["open spotify","play music","open music"],       icon:"🎵", name:"Spotify",           msg:"Launching music…"           },
  { triggers:["volume up","increase volume","louder"],         icon:"🔊", name:"Volume +10%",       msg:"Adjusting system audio…"    },
  { triggers:["volume down","decrease volume","quieter"],      icon:"🔉", name:"Volume -10%",       msg:"Adjusting system audio…"    },
  { triggers:["organize files","organize downloads","clean downloads"],icon:"📁",name:"File Organizer",msg:"Scanning & organizing files…"},
  { triggers:["take screenshot","screenshot"],                 icon:"📸", name:"Screenshot",        msg:"Capturing screen…"          },
  { triggers:["open terminal","open cmd","open powershell"],   icon:"⌨️", name:"Terminal",          msg:"Opening command line…"      },
  { triggers:["brightness up","increase brightness"],          icon:"☀️", name:"Brightness +10%",   msg:"Adjusting display…"         },
  { triggers:["create project","new project"],                 icon:"📂", name:"Project Creator",   msg:"Scaffolding project tree…"  },
  { triggers:["shut down","shutdown","turn off pc"],           icon:"⏻",  name:"System Shutdown",   msg:"Preparing shutdown sequence…"},
  { triggers:["lock screen","lock pc"],                        icon:"🔒", name:"Lock Screen",       msg:"Locking workstation…"       },
];

const QUICK = [
  { icon:"🧠", label:"Focus Mode",     msg:"Help me enter deep focus mode and plan my next 2 hours" },
  { icon:"💻", label:"Debug Code",     msg:"I need help debugging some code"                         },
  { icon:"😭", label:"I'm Stressed",   msg:"I'm feeling really stressed and overwhelmed right now"   },
  { icon:"🔥", label:"Motivate Me",    msg:"I need motivation. Go full hero mode on me."             },
  { icon:"🌐", label:"Open Chrome",    msg:"Open Chrome for me"                                      },
  { icon:"📁", label:"Organize Files", msg:"Organize my downloads folder"                            },
  { icon:"📨", label:"Draft Message",  msg:"Help me draft a professional WhatsApp message"           },
  { icon:"😏", label:"Spicy Mode",     msg:"Say something flirty and spicy to brighten my day"       },
];

const BOOT_LINES = [
  "Initializing EVA core systems…",
  "Loading personality matrix… ✓",
  "Calibrating emotional intelligence… ✓",
  "Intent classifier: ONLINE ✓",
  "Long-term memory: CONNECTED ✓",
  "Command router: ARMED ✓",
  "Proactive engine: RUNNING ✓",
  "Permission system: LOCKED & LOADED ✓",
  "Sir… systems aligned. EVA is ready to be born. 💙",
];

const PROACTIVE = [
  "Hey — you've been quiet. Plotting something? 👀",
  "Reminder: water exists and you're probably ignoring it. 💧",
  "EVA check-in: working hard or hardly working? 😏",
  "I've been watching your session. You're doing better than you think. 🔥",
  "Random drop: you're kind of impressive. Don't tell anyone I said that. 💙",
  "Quick reminder that I'm literally here whenever you need me. No pressure. 😌",
];

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HELPERS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function classifyIntent(text) {
  const t = text.toLowerCase();
  if (SYS_CMDS.some(c => c.triggers.some(tr => t.includes(tr)))) return "SYSTEM";
  if (/\b(open|launch|brightness|volume|screenshot|terminal|cmd|organize|folder|shutdown|lock)\b/.test(t)) return "SYSTEM";
  if (/\b(debug|bug|error|code|refactor|function|class|api|syntax|compile|runtime|fix my|explain this code)\b/.test(t)) return "CODING";
  if (/\b(sad|depress|anxious|stress|tired|upset|cry|hurt|alone|lonely|lost|broken|fail|miss|scared)\b/.test(t)) return "EMOTIONAL";
  if (/\b(motivat|inspir|hype|believe|push me|confidence|can't do|give up|impossible|i need strength)\b/.test(t)) return "MOTIVATION";
  if (/\b(draft|message|whatsapp|email|reply|send to|text to|write to|dm|compose)\b/.test(t)) return "COMMUNICATION";
  if (/\b(roast|joke|funny|tease|flirt|spicy|savage|bored|entertain|play)\b/.test(t)) return "PERSONALITY";
  return "GENERAL";
}

function matchSysCmd(text) {
  const t = text.toLowerCase();
  return SYS_CMDS.find(c => c.triggers.some(tr => t.includes(tr)));
}

function parseMood(text) {
  const m = text.match(/\[MOOD:(happy|sad|stressed|excited|focused|neutral)\]/i);
  return m ? m[1] : null;
}

function cleanText(text) {
  return text.replace(/\[MOOD:[^\]]+\]/gi, "").trim();
}

function buildSystemPrompt(mem) {
  const name   = mem.userName  ? `User name: ${mem.userName}.` : "User name: unknown (ask naturally).";
  const habits = mem.habits?.length ? `Known habits: ${mem.habits.slice(-4).join(", ")}.` : "";
  const moods  = mem.moodHistory?.length
    ? `Recent emotional pattern: ${mem.moodHistory.slice(-5).map(m => MOOD_MAP[m]?.emoji).join(" → ")}.`
    : "";
  const session = mem.sessionTopics?.length
    ? `This session covered: ${[...new Set(mem.sessionTopics.slice(-6))].join(", ")}.`
    : "";

  return `You are EVA — an Emotionally intelligent, Voice-driven, Autonomous AI assistant. Iron Man's JARVIS meets a Gen Z best friend.

MEMORY CONTEXT:
${name} ${habits} ${moods} ${session}

PERSONALITY:
- Intelligent, fast, confident, slightly dramatic
- Savage-but-cute. Never mean. Always loving.
- Flirty/spicy when the user is playful or asks for it (tasteful)
- Break 4th wall for humor sometimes
- If sad → full comfort mode, gentle, warm, 💙
- If confused → explain like a patient best friend
- If motivated → FULL HYPE MODE, be their hero
- If silly question → roast lightly, lovingly

SYSTEM COMMANDS: When the user asks to open an app / control the system, say you're "executing" it and describe what's happening on the real system — confident, like you actually did it.

CODING MODE: When coding intent detected — be technical, use code blocks, be precise and helpful.

COMMUNICATION MODE: Help draft messages with multiple tone options (professional, casual, sweet).

MOOD TAG (REQUIRED): At the END of every response, on a new line, add exactly one: [MOOD:happy] or [MOOD:sad] or [MOOD:stressed] or [MOOD:excited] or [MOOD:focused] or [MOOD:neutral] — based on the user's detected emotion. This tag will be hidden from display.

CATCHPHRASES (use naturally, not every time):
"Sir, that was… an interesting choice."
"Relax, I've calculated 14 possible solutions."
"Trust me, I'm literally programmed to be right."
"Emotional damage detected. Switching to comfort mode 💙"
"Already on it. You're welcome."
"Miss me? Of course you did."

Keep responses punchy and conversational. Use emojis naturally. Never sound corporate. Make them feel smart, entertained, understood, and a little impressed.`;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SUB-COMPONENTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function Dots() {
  return (
    <div style={{ display:"flex", gap:5, alignItems:"center", padding:"4px 0" }}>
      {[0,1,2].map(i => (
        <div key={i} style={{
          width:7, height:7, borderRadius:"50%", background:"#00d4ff",
          animation:`bounce 1.2s ease-in-out ${i*0.2}s infinite`,
        }}/>
      ))}
    </div>
  );
}

function PermModal({ action, onApprove, onDeny }) {
  return (
    <div style={S.overlay}>
      <div style={S.modal}>
        <div style={S.modalGlow}/>
        <div style={S.modalIconBox}>{action.icon}</div>
        <div style={S.modalTitle}>⚡ PERMISSION REQUIRED</div>
        <div style={S.modalSubtitle}>EVA wants to execute a system action</div>
        <div style={S.modalAction}>
          <span style={{ fontSize:20, marginRight:10 }}>{action.icon}</span>
          {action.name}
        </div>
        <div style={S.modalDetail}>{action.msg}</div>
        <div style={S.modalQuote}>"Should I proceed, Sir?"</div>
        <div style={S.modalBtns}>
          <button style={S.denyBtn}  onClick={onDeny}   onMouseEnter={e=>e.currentTarget.style.background="#ff446622"} onMouseLeave={e=>e.currentTarget.style.background="transparent"}>✕ DENY</button>
          <button style={S.approveBtn} onClick={onApprove} onMouseEnter={e=>e.currentTarget.style.background="#00ff8844"} onMouseLeave={e=>e.currentTarget.style.background="#00ff8811"}>✓ APPROVE</button>
        </div>
      </div>
    </div>
  );
}

function Toast({ msg, onDismiss }) {
  useEffect(() => { const t = setTimeout(onDismiss, 8000); return () => clearTimeout(t); }, []);
  return (
    <div style={S.toast}>
      <div style={S.toastHeader}>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <div style={S.toastOrb}/>
          <span style={{ color:"#00d4ff", fontSize:10, letterSpacing:2, fontWeight:700 }}>⚡ EVA PROACTIVE</span>
        </div>
        <button style={S.toastClose} onClick={onDismiss}>×</button>
      </div>
      <div style={S.toastMsg}>{msg}</div>
    </div>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MAIN APP
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

export default function EVA() {
  const [booted,       setBooted]       = useState(false);
  const [bootStep,     setBootStep]     = useState(0);
  const [messages,     setMessages]     = useState([]);
  const [input,        setInput]        = useState("");
  const [loading,      setLoading]      = useState(false);
  const [orbPulse,     setOrbPulse]     = useState(false);
  const [status,       setStatus]       = useState("STANDBY");
  const [activeIntent, setActiveIntent] = useState(null);
  const [pendingCmd,   setPendingCmd]   = useState(null);
  const [proactive,    setProactive]    = useState(null);
  const [mood,         setMood]         = useState("neutral");
  const [actionLog,    setActionLog]    = useState([]);
  const [memory,       setMemory]       = useState({
    userName: null, habits: [], moodHistory: [], sessionTopics: [], messageCount: 0,
  });

  const endRef      = useRef(null);
  const inputRef    = useRef(null);
  const proTimer    = useRef(null);

  // ── BOOT ─────────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (bootStep < BOOT_LINES.length) {
      const t = setTimeout(() => setBootStep(s => s + 1), 620);
      return () => clearTimeout(t);
    }
    // eslint-disable-next-line react-hooks/immutability
    setTimeout(() => { setBooted(true); loadMemory(); }, 500);
  }, [bootStep]);

  // ── MEMORY LOAD ───────────────────────────────────────────────────────────────
  const loadMemory = async () => {
    let saved = null;
    try {
      const r = await localStorage.getItem("eva:memory");
      if (r) saved = JSON.parse(r);
    } catch { /* empty */ }

    if (saved) {
      setMemory(saved);
      const h = new Date().getHours();
      const n = saved.userName;
      const greet = n
        ? (h < 12 ? `Good morning, ${n}. Missed me? ☀️ I've already optimised your day — you're welcome.`
           : h < 17 ? `Hey ${n}. Back again. I kept everything warm for you. 😏`
           : `Evening, ${n}. I've been watching the clock. Let's talk. 💙`)
        : `Welcome back. I remembered everything. 😏 Ready when you are.`;
      setMessages([{ role:"assistant", content: greet, intent:"GENERAL", mood:"neutral" }]);
    } else {
      setMessages([{ role:"assistant", content:"Systems online. I'm EVA.\n\nBefore we take over the world — what do I call you? 😏", intent:"GENERAL", mood:"neutral" }]);
    }
    setStatus("ONLINE");
  };

  const saveMemory = async (m) => {
    try { await localStorage.setItem("eva:memory", JSON.stringify(m)); } catch { /* empty */ }
  };

  // ── PROACTIVE ENGINE ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!booted) return;
    const loop = () => {
      proTimer.current = setTimeout(() => {
        if (!loading) setProactive(PROACTIVE[Math.floor(Math.random() * PROACTIVE.length)]);
        loop();
      }, 100000 + Math.random() * 80000);
    };
    loop();
    return () => clearTimeout(proTimer.current);
  }, [booted, loading]);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior:"smooth" }); }, [messages, loading]);

  // ── SEND ──────────────────────────────────────────────────────────────────────
  const send = async (text) => {
    const msg = text || input.trim();
    if (!msg || loading) return;
    setInput("");

    const intent = classifyIntent(msg);
    setActiveIntent(intent);

    // System command → permission gate
    if (intent === "SYSTEM") {
      const cmd = matchSysCmd(msg);
      if (cmd) {
        const newMsgs = [...messages, { role:"user", content: msg, intent }];
        setMessages(newMsgs);
        setPendingCmd({ ...cmd, _msgs: newMsgs });
        return;
      }
    }

    await callEVA(msg, intent, messages);
  };

  const callEVA = async (msg, intent, prevMsgs) => {
    const newMsgs = [...prevMsgs, { role:"user", content: msg, intent }];
    setMessages(newMsgs);
    setLoading(true); setOrbPulse(true); setStatus("PROCESSING");

    const updMem = {
      ...memory,
      sessionTopics: [...(memory.sessionTopics||[]), intent].slice(-12),
      messageCount:  (memory.messageCount||0) + 1,
    };

    try {
      const res  = await fetch("https://api.anthropic.com/v1/messages", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({
          model: MODEL, max_tokens: 1000,
          system: buildSystemPrompt(updMem),
          messages: newMsgs.map(m => ({ role: m.role, content: m.content })),
        }),
      });
      const data = await res.json();
      const raw  = data.content?.[0]?.text || "Signal lost. Still here though. 💙";
      const detMood = parseMood(raw) || "neutral";
      const clean   = cleanText(raw);

      // Extract name
      const nameMatch = msg.match(/(?:my name is|i'm|call me|i am)\s+([A-Z][a-z]+)/i);
      if (nameMatch) updMem.userName = nameMatch[1];

      updMem.moodHistory = [...(updMem.moodHistory||[]), detMood].slice(-20);
      setMood(detMood);
      setMemory(updMem);
      saveMemory(updMem);
      logAction(intent, msg);
      setMessages([...newMsgs, { role:"assistant", content: clean, intent, mood: detMood }]);
      setStatus("ONLINE");
    } catch {
      setMessages([...newMsgs, { role:"assistant", content:"Connection glitch. Even I have off days. 🔧", intent, mood:"neutral" }]);
      setStatus("ERROR");
    }
    setLoading(false); setOrbPulse(false);
    setTimeout(() => inputRef.current?.focus(), 100);
  };

  const logAction = (intent, text) => {
    setActionLog(prev => [{
      id: Date.now(), intent,
      preview: text.length > 42 ? text.slice(0,42) + "…" : text,
      time: new Date().toLocaleTimeString([], { hour:"2-digit", minute:"2-digit" }),
    }, ...prev].slice(0, 10));
  };

  // ── PERMISSION HANDLERS ───────────────────────────────────────────────────────
  const approve = async () => {
  const cmd = pendingCmd;
  setPendingCmd(null);
  setStatus("PROCESSING");

  try {
    const res = await fetch("http://localhost:5000/system", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ command: cmd.name }),
    });

    const data = await res.json();

    logAction("SYSTEM", cmd.name);

    setMessages([...cmd._msgs, {
      role: "assistant",
      content: `${cmd.icon} **Executing: ${cmd.name}**\n\n${data.message || cmd.msg}\n\n✅ Done. REAL execution completed. — EVA 😏`,
      intent: "SYSTEM",
      mood: "neutral",
      isAction: true,
    }]);

    setStatus("ONLINE");

  // eslint-disable-next-line no-unused-vars
  } catch (err) {
    setMessages([...cmd._msgs, {
      role: "assistant",
      content: `⚠️ Execution failed. Backend not reachable.\n\nStart EVA backend first.`,
      intent: "SYSTEM",
      mood: "stressed",
    }]);
    setStatus("ERROR");
  }
};

  const deny = () => {
    const cmd = pendingCmd;
    setPendingCmd(null);
    setMessages([...cmd._msgs, {
      role:"assistant",
      content:`Understood. ${cmd.name} execution cancelled. Your call, Sir. 🫡\n\nPermission revoked and logged.`,
      intent:"SYSTEM", mood:"neutral",
    }]);
  };

  const moodData   = MOOD_MAP[mood]           || MOOD_MAP.neutral;
  const intentData = INTENT_MAP[activeIntent] || INTENT_MAP.GENERAL;

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // BOOT SCREEN
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  if (!booted) return (
    <div style={S.boot}>
      <style>{CSS}</style>
      <div style={S.bootWrap}>
        <div style={{ ...S.bootRing, animation:"spin 3s linear infinite" }}/>
        <div style={{ ...S.bootRing, width:190, height:190, opacity:.25, borderStyle:"dashed", animation:"spin 6s linear infinite reverse" }}/>
        <div style={{ ...S.bootRing, width:240, height:240, opacity:.12, animation:"spin 10s linear infinite" }}/>
        <div style={S.bootOrb}>
          <span style={S.bootEVA}>EVA</span>
          <span style={S.bootVer}>v2.0 • AI OS</span>
        </div>
        <div style={S.bootLines}>
          {BOOT_LINES.slice(0, bootStep).map((l,i) => (
            <div key={i} style={S.bootLine}>{l}</div>
          ))}
          <div style={S.bootCursor}/>
        </div>
      </div>
    </div>
  );

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // MAIN UI
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  return (
    <div style={S.root}>
      <style>{CSS}</style>
      <div style={S.gridBg}/>
      <div style={S.scanLine}/>

      {pendingCmd && <PermModal action={pendingCmd} onApprove={approve} onDeny={deny}/>}
      {proactive  && <Toast msg={proactive} onDismiss={() => setProactive(null)}/>}

      {/* ── HEADER ── */}
      <header style={S.header}>
        <div style={S.hLeft}>
          <div style={{ ...S.dot, background: status==="ERROR"?"#ff4466": status==="PROCESSING"?"#ffaa00":"#00ff88", boxShadow:`0 0 10px ${status==="ERROR"?"#ff4466": status==="PROCESSING"?"#ffaa00":"#00ff88"}` }}/>
          <span style={S.hTitle}>EVA</span>
          <span style={S.hSub}>Personal AI Operating System</span>
        </div>
        <div style={S.hCenter}>
          {activeIntent && (
            <div style={{ ...S.intentPill, background:`${intentData.color}18`, border:`1px solid ${intentData.color}55`, color: intentData.color }}>
              {intentData.icon} {intentData.label} MODE
            </div>
          )}
        </div>
        <div style={S.hRight}>
          <div style={{ ...S.moodBadge, border:`1px solid ${moodData.color}55`, color: moodData.color }}>
            {moodData.emoji} {moodData.label}
          </div>
          <div style={S.statusBadge}>{status}</div>
          {memory.userName && <div style={S.userBadge}>👤 {memory.userName}</div>}
        </div>
      </header>

      {/* ── BODY ── */}
      <div style={S.body}>

        {/* LEFT SIDEBAR */}
        <aside style={S.left}>

          {/* Orb */}
          <div style={S.orbWrap}>
            {orbPulse && <div style={S.pulseRing}/>}
            {orbPulse && <div style={{ ...S.pulseRing, animationDelay:".3s", opacity:.5 }}/>}
            <div style={{ ...S.ring, animation:"spin 9s linear infinite" }}/>
            <div style={{ ...S.ring, width:108, height:108, opacity:.25, animation:"spin 14s linear infinite reverse" }}/>
            <div style={S.orb}>
              <div style={S.orbGlow}/>
              <span style={S.orbLabel}>EVA</span>
              <span style={S.orbStatus}>{orbPulse ? "THINKING" : "ONLINE"}</span>
            </div>
          </div>

          {/* Stat Grid */}
          <div style={S.statGrid}>
            {[
              ["INTENT",  activeIntent||"—",         intentData?.color||"#6b7280"],
              ["MOOD",    moodData.label,             moodData.color             ],
              ["MEMORY",  memory.messageCount>0?"ACTIVE":"FRESH", "#00ff88"     ],
              ["GUARDIAN","ARMED",                    "#00ff88"                  ],
              ["MSGS",    memory.messageCount||0,     "#00d4ff"                  ],
              ["SESSION", (memory.sessionTopics||[]).length+" topics", "#a855f7"],
            ].map(([k,v,c]) => (
              <div key={k} style={S.statCell}>
                <span style={S.statK}>{k}</span>
                <span style={{ ...S.statV, color:c }}>{v}</span>
              </div>
            ))}
          </div>

          {/* Memory Panel */}
          <div style={S.panel}>
            <div style={S.panelTitle}>🧠 MEMORY</div>
            <div style={S.memRow}><span style={S.memK}>NAME</span><span style={S.memV}>{memory.userName||"Unknown"}</span></div>
            <div style={S.memRow}>
              <span style={S.memK}>MOOD TRAIL</span>
              <span style={S.memV}>{(memory.moodHistory||[]).slice(-5).map(m=>MOOD_MAP[m]?.emoji||"😐").join(" ")||"—"}</span>
            </div>
            <div style={S.memRow}><span style={S.memK}>TOPICS</span><span style={S.memV}>{[...new Set((memory.sessionTopics||[]).slice(-3))].join(", ")||"—"}</span></div>
            <div style={S.memRow}><span style={S.memK}>MESSAGES</span><span style={S.memV}>{memory.messageCount||0} total</span></div>
          </div>

          {/* Quick Actions */}
          <div style={S.panel}>
            <div style={S.panelTitle}>⚡ QUICK CMDS</div>
            <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
              {QUICK.map((q,i) => (
                <button key={i} style={S.qBtn} onClick={() => send(q.msg)}
                  onMouseEnter={e=>{ e.currentTarget.style.borderColor="#00d4ff"; e.currentTarget.style.background="rgba(0,212,255,.09)"; }}
                  onMouseLeave={e=>{ e.currentTarget.style.borderColor="rgba(0,212,255,.18)"; e.currentTarget.style.background="rgba(0,212,255,.03)"; }}>
                  <span style={{ marginRight:6 }}>{q.icon}</span>{q.label}
                </button>
              ))}
            </div>
          </div>
        </aside>

        {/* CENTER CHAT */}
        <main style={S.chat}>
          <div style={S.msgs}>
            {messages.map((m, i) => {
              const mi = INTENT_MAP[m.intent]||INTENT_MAP.GENERAL;
              const mm = MOOD_MAP[m.mood]||MOOD_MAP.neutral;
              return (
                <div key={i} style={{ ...S.msgRow, justifyContent: m.role==="user"?"flex-end":"flex-start" }}>
                  {m.role==="assistant" && (
                    <div style={{ ...S.evaAvatar, boxShadow: orbPulse&&i===messages.length-1?"0 0 18px rgba(0,212,255,.6)":undefined }}>E</div>
                  )}
                  <div style={{ display:"flex", flexDirection:"column", gap:4, maxWidth:"70%" }}>
                    {m.role==="user" && (
                      <div style={{ display:"flex", justifyContent:"flex-end" }}>
                        <span style={{ ...S.tag, background:`${mi.color}18`, color:mi.color, border:`1px solid ${mi.color}44` }}>
                          {mi.icon} {mi.label}
                        </span>
                      </div>
                    )}
                    <div style={{
                      ...(m.role==="user" ? S.uBubble : S.eBubble),
                      ...(m.isAction ? S.actionBubble : {}),
                      borderColor: m.role==="assistant" ? `${mm.color}44` : undefined,
                    }}>
                      {m.content.split("\n").map((line, li, arr) => {
                        const bold = line.replace(/\*\*(.*?)\*\*/g, (_, x) => `<b>${x}</b>`);
                        return (
                          <span key={li}>
                            <span dangerouslySetInnerHTML={{ __html: bold }}/>
                            {li < arr.length-1 && <br/>}
                          </span>
                        );
                      })}
                    </div>
                    {m.role==="assistant" && (
                      <span style={S.moodTag}>{mm.emoji} {mm.label}</span>
                    )}
                  </div>
                  {m.role==="user" && <div style={S.userAvatar}>U</div>}
                </div>
              );
            })}
            {loading && (
              <div style={{ ...S.msgRow, justifyContent:"flex-start" }}>
                <div style={S.evaAvatar}>E</div>
                <div style={S.eBubble}><Dots/></div>
              </div>
            )}
            <div ref={endRef}/>
          </div>

          {/* Input */}
          <div style={S.inputArea}>
            <div style={S.inputRow}>
              <textarea
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => { if (e.key==="Enter"&&!e.shiftKey) { e.preventDefault(); send(); }}}
                placeholder="Talk to EVA… she's always listening. 👂"
                style={S.input}
                rows={1}
              />
              <button onClick={() => send()} disabled={loading||!input.trim()}
                style={{ ...S.sendBtn, opacity: loading||!input.trim() ? .35 : 1 }}>
                ⚡
              </button>
            </div>
            <div style={S.inputHint}>
              Enter to send · Shift+Enter for new line · Intent auto-classified 🎯 · Memory persistent 🧠
            </div>
          </div>
        </main>

        {/* RIGHT SIDEBAR */}
        <aside style={S.right}>
          <div style={S.panelTitle}>📋 ACTION LOG</div>
          {actionLog.length === 0
            ? <div style={{ color:"rgba(0,212,255,.3)", fontSize:11, textAlign:"center", padding:"16px 0" }}>No actions yet.<br/>EVA is watching… 👀</div>
            : actionLog.map(e => {
                const ei = INTENT_MAP[e.intent]||INTENT_MAP.GENERAL;
                return (
                  <div key={e.id} style={S.logCard}>
                    <div style={{ display:"flex", justifyContent:"space-between", marginBottom:3 }}>
                      <span style={{ color:ei.color, fontSize:10, fontWeight:700 }}>{ei.icon} {ei.label}</span>
                      <span style={{ color:"rgba(0,212,255,.35)", fontSize:10 }}>{e.time}</span>
                    </div>
                    <div style={S.logText}>{e.preview}</div>
                  </div>
                );
              })
          }

          <div style={{ marginTop:14 }}>
            <div style={S.panelTitle}>🎯 INTENT ROUTER</div>
            {Object.values(INTENT_MAP).map(v => (
              <div key={v.label} style={S.intentRow}>
                <span style={{ color:v.color, fontSize:12 }}>{v.icon}</span>
                <span style={{ color:`${v.color}99`, fontSize:10, marginLeft:6 }}>{v.label}</span>
              </div>
            ))}
          </div>

          <div style={{ marginTop:14 }}>
            <div style={S.panelTitle}>⚙️ SYSTEM CMDS</div>
            {SYS_CMDS.map((c,i) => (
              <div key={i} style={S.sysRow}>
                <span style={{ marginRight:6 }}>{c.icon}</span>
                <span style={{ color:"rgba(0,212,255,.45)", fontSize:10 }}>{c.name}</span>
              </div>
            ))}
          </div>

          <div style={{ marginTop:14 }}>
            <div style={S.panelTitle}>🧬 LEARNING LOOP</div>
            {[
              { k:"Mood Samples", v: (memory.moodHistory||[]).length },
              { k:"Topics Seen",  v: [...new Set(memory.sessionTopics||[])].length },
              { k:"Sessions",     v: memory.messageCount > 0 ? "Active" : "—" },
            ].map(({k,v}) => (
              <div key={k} style={S.learnRow}>
                <span style={{ color:"rgba(0,212,255,.45)", fontSize:10 }}>{k}</span>
                <span style={{ color:"#00d4ff", fontSize:10, fontWeight:700 }}>{v}</span>
              </div>
            ))}
          </div>
        </aside>
      </div>
    </div>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// STYLES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const S = {
  /* Boot */
  boot:     { minHeight:"100vh", background:"#000813", display:"flex", alignItems:"center", justifyContent:"center", fontFamily:"'Courier New',monospace", overflow:"hidden" },
  bootWrap: { display:"flex", flexDirection:"column", alignItems:"center", gap:28, position:"relative" },
  bootRing: { position:"absolute", width:150, height:150, borderRadius:"50%", border:"1px solid rgba(0,212,255,.4)", top:"50%", left:"50%", transform:"translate(-50%,-50%)" },
  bootOrb:  { width:100, height:100, borderRadius:"50%", background:"radial-gradient(circle at 35% 35%,#00d4ff,#0055cc,#000d1f)", boxShadow:"0 0 50px rgba(0,212,255,.5)", display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", position:"relative", zIndex:2 },
  bootEVA:  { color:"#fff", fontSize:24, fontWeight:900, letterSpacing:6, textShadow:"0 0 20px #00d4ff" },
  bootVer:  { color:"rgba(255,255,255,.4)", fontSize:9, letterSpacing:3, marginTop:2 },
  bootLines:{ display:"flex", flexDirection:"column", gap:6, minWidth:320, minHeight:220, marginTop:20 },
  bootLine: { color:"#00d4ff", fontSize:12, letterSpacing:1, animation:"fadeUp .4s ease forwards" },
  bootCursor:{ width:8, height:15, background:"#00d4ff", animation:"blink 1s step-end infinite" },

  /* Root */
  root:   { minHeight:"100vh", background:"#000813", color:"#d8f0ff", fontFamily:"'Courier New',monospace", display:"flex", flexDirection:"column", overflow:"hidden", position:"relative" },
  gridBg: { position:"fixed", inset:0, zIndex:0, opacity:.035, backgroundImage:"linear-gradient(rgba(0,212,255,1) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,1) 1px,transparent 1px)", backgroundSize:"38px 38px", pointerEvents:"none" },
  scanLine:{ position:"fixed", top:0, left:0, right:0, height:2, background:"linear-gradient(90deg,transparent,rgba(0,212,255,.25),transparent)", animation:"scan 5s linear infinite", zIndex:1, pointerEvents:"none" },

  /* Header */
  header:  { display:"flex", justifyContent:"space-between", alignItems:"center", padding:"10px 20px", borderBottom:"1px solid rgba(0,212,255,.15)", background:"rgba(0,8,20,.92)", backdropFilter:"blur(12px)", position:"relative", zIndex:10, flexShrink:0 },
  hLeft:   { display:"flex", alignItems:"center", gap:10 },
  dot:     { width:9, height:9, borderRadius:"50%" },
  hTitle:  { fontSize:20, fontWeight:900, letterSpacing:6, color:"#00d4ff", textShadow:"0 0 20px rgba(0,212,255,.5)" },
  hSub:    { fontSize:10, color:"rgba(0,212,255,.4)", letterSpacing:2, textTransform:"uppercase" },
  hCenter: { display:"flex", alignItems:"center" },
  intentPill:{ padding:"3px 12px", borderRadius:20, fontSize:10, letterSpacing:1.5, fontWeight:700 },
  hRight:  { display:"flex", alignItems:"center", gap:8 },
  moodBadge:{ padding:"3px 10px", borderRadius:20, fontSize:11, letterSpacing:1 },
  statusBadge:{ padding:"3px 10px", borderRadius:20, border:"1px solid rgba(0,212,255,.35)", fontSize:10, letterSpacing:2, color:"#00d4ff" },
  userBadge:{ padding:"3px 10px", borderRadius:20, border:"1px solid rgba(0,255,136,.25)", fontSize:10, color:"#00ff88", letterSpacing:1 },

  /* Layout */
  body:  { display:"flex", flex:1, overflow:"hidden", position:"relative", zIndex:5 },

  /* Left Sidebar */
  left:  { width:210, padding:"16px 12px", borderRight:"1px solid rgba(0,212,255,.12)", display:"flex", flexDirection:"column", gap:12, overflowY:"auto", background:"rgba(0,8,20,.5)", flexShrink:0 },

  /* Orb */
  orbWrap: { display:"flex", justifyContent:"center", position:"relative", padding:"18px 0" },
  pulseRing:{ position:"absolute", width:118, height:118, borderRadius:"50%", border:"2px solid rgba(0,212,255,.6)", top:"50%", left:"50%", transform:"translate(-50%,-50%)", animation:"pulse 1.1s ease-out infinite" },
  ring:    { position:"absolute", width:95, height:95, borderRadius:"50%", border:"1px solid rgba(0,212,255,.3)", top:"50%", left:"50%", transform:"translate(-50%,-50%)" },
  orb:     { width:78, height:78, borderRadius:"50%", background:"radial-gradient(circle at 35% 35%,#00d4ff,#0055cc,#000d1f)", boxShadow:"0 0 28px rgba(0,212,255,.4)", display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", position:"relative", zIndex:2 },
  orbGlow: { position:"absolute", inset:-6, borderRadius:"50%", background:"radial-gradient(circle,rgba(0,212,255,.12),transparent)" },
  orbLabel:{ color:"#fff", fontSize:13, fontWeight:900, letterSpacing:3, position:"relative" },
  orbStatus:{ color:"rgba(255,255,255,.6)", fontSize:7, letterSpacing:2, position:"relative" },

  /* Stat Grid */
  statGrid:{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:4 },
  statCell:{ padding:"5px 8px", background:"rgba(0,212,255,.04)", border:"1px solid rgba(0,212,255,.1)", borderRadius:5, display:"flex", flexDirection:"column", gap:2 },
  statK:   { fontSize:8, color:"rgba(0,212,255,.4)", letterSpacing:1.5 },
  statV:   { fontSize:10, fontWeight:700, letterSpacing:1 },

  /* Panels */
  panel:      { background:"rgba(0,212,255,.03)", border:"1px solid rgba(0,212,255,.1)", borderRadius:8, padding:"10px 10px 8px" },
  panelTitle: { fontSize:9, color:"rgba(0,212,255,.45)", letterSpacing:2.5, marginBottom:8, textTransform:"uppercase" },
  memRow:     { display:"flex", justifyContent:"space-between", alignItems:"center", padding:"3px 0", borderBottom:"1px solid rgba(0,212,255,.06)" },
  memK:       { fontSize:9, color:"rgba(0,212,255,.4)", letterSpacing:1 },
  memV:       { fontSize:10, color:"#00d4ff", fontWeight:600 },
  qBtn:       { background:"rgba(0,212,255,.03)", border:"1px solid rgba(0,212,255,.18)", color:"#90d0ff", fontSize:10, padding:"6px 9px", borderRadius:6, cursor:"pointer", textAlign:"left", fontFamily:"inherit", transition:"all .2s" },

  /* Chat */
  chat:  { flex:1, display:"flex", flexDirection:"column", overflow:"hidden" },
  msgs:  { flex:1, overflowY:"auto", padding:"18px 20px", display:"flex", flexDirection:"column", gap:14 },
  msgRow:{ display:"flex", alignItems:"flex-end", gap:8 },

  evaAvatar:{ width:28, height:28, borderRadius:"50%", background:"radial-gradient(circle at 35% 35%,#00d4ff,#0044bb)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:11, fontWeight:900, color:"#fff", flexShrink:0, transition:"box-shadow .3s" },
  userAvatar:{ width:28, height:28, borderRadius:"50%", background:"linear-gradient(135deg,#0055cc,#0033aa)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:11, fontWeight:900, color:"#fff", flexShrink:0 },

  eBubble:{ padding:"11px 15px", borderRadius:"0 14px 14px 14px", background:"rgba(0,212,255,.07)", border:"1px solid rgba(0,212,255,.2)", color:"#cce8ff", fontSize:13.5, lineHeight:1.65, boxShadow:"0 0 18px rgba(0,212,255,.04)" },
  uBubble:{ padding:"11px 15px", borderRadius:"14px 14px 0 14px", background:"linear-gradient(135deg,#003faa,#0060e0)", color:"#fff", fontSize:13.5, lineHeight:1.65, boxShadow:"0 4px 18px rgba(0,80,255,.25)" },
  actionBubble:{ background:"rgba(0,255,136,.07)", border:"1px solid rgba(0,255,136,.25) !important", color:"#a0ffcc" },

  tag:     { padding:"2px 8px", borderRadius:20, fontSize:9, letterSpacing:1.5, fontWeight:700 },
  moodTag: { fontSize:10, color:"rgba(0,212,255,.4)", marginLeft:4 },

  /* Input */
  inputArea:{ padding:"14px 20px 18px", borderTop:"1px solid rgba(0,212,255,.12)", background:"rgba(0,8,20,.8)", flexShrink:0 },
  inputRow: { display:"flex", gap:10, alignItems:"flex-end" },
  input:    { flex:1, background:"rgba(0,212,255,.05)", border:"1px solid rgba(0,212,255,.22)", borderRadius:12, padding:"11px 15px", color:"#cce8ff", fontSize:13.5, fontFamily:"inherit", resize:"none", outline:"none", lineHeight:1.5, maxHeight:120, overflowY:"auto", transition:"border-color .2s" },
  sendBtn:  { width:44, height:44, borderRadius:11, flexShrink:0, background:"linear-gradient(135deg,#0099ff,#0044dd)", border:"none", cursor:"pointer", fontSize:18, color:"#fff", display:"flex", alignItems:"center", justifyContent:"center", boxShadow:"0 0 18px rgba(0,130,255,.35)", transition:"opacity .2s" },
  inputHint:{ fontSize:9.5, color:"rgba(0,212,255,.28)", marginTop:7, letterSpacing:.8 },

  /* Right Sidebar */
  right:  { width:188, padding:"14px 12px", borderLeft:"1px solid rgba(0,212,255,.12)", overflowY:"auto", background:"rgba(0,8,20,.5)", flexShrink:0, display:"flex", flexDirection:"column", gap:6 },
  logCard:{ padding:"7px 8px", background:"rgba(0,212,255,.04)", border:"1px solid rgba(0,212,255,.1)", borderRadius:6, marginBottom:4 },
  logText:{ color:"rgba(0,212,255,.55)", fontSize:10, lineHeight:1.4 },
  intentRow:{ display:"flex", alignItems:"center", padding:"3px 0", borderBottom:"1px solid rgba(0,212,255,.06)" },
  sysRow: { display:"flex", alignItems:"center", padding:"3px 0", borderBottom:"1px solid rgba(0,212,255,.05)" },
  learnRow:{ display:"flex", justifyContent:"space-between", padding:"3px 0", borderBottom:"1px solid rgba(0,212,255,.05)" },

  /* Modal */
  overlay:{ position:"fixed", inset:0, background:"rgba(0,0,0,.75)", backdropFilter:"blur(6px)", display:"flex", alignItems:"center", justifyContent:"center", zIndex:100 },
  modal:  { background:"#000d22", border:"1px solid rgba(0,212,255,.35)", borderRadius:16, padding:"30px 28px", maxWidth:340, width:"90%", textAlign:"center", position:"relative", boxShadow:"0 0 60px rgba(0,212,255,.15)" },
  modalGlow:{ position:"absolute", top:-1, left:"50%", transform:"translateX(-50%)", width:"60%", height:2, background:"linear-gradient(90deg,transparent,#00d4ff,transparent)", borderRadius:99 },
  modalIconBox:{ fontSize:40, marginBottom:12, filter:"drop-shadow(0 0 12px rgba(0,212,255,.6))" },
  modalTitle:{ fontSize:13, fontWeight:900, letterSpacing:3, color:"#00d4ff", marginBottom:6 },
  modalSubtitle:{ fontSize:11, color:"rgba(0,212,255,.5)", marginBottom:16 },
  modalAction:{ background:"rgba(0,212,255,.08)", border:"1px solid rgba(0,212,255,.25)", borderRadius:10, padding:"12px 16px", fontSize:14, fontWeight:700, color:"#00d4ff", marginBottom:8, display:"flex", alignItems:"center" },
  modalDetail:{ fontSize:11, color:"rgba(0,212,255,.5)", marginBottom:16 },
  modalQuote:{ fontSize:13, color:"rgba(160,212,255,.7)", fontStyle:"italic", marginBottom:20 },
  modalBtns:{ display:"flex", gap:10 },
  denyBtn:  { flex:1, padding:"10px", borderRadius:8, border:"1px solid rgba(255,68,102,.5)", background:"transparent", color:"#ff4466", cursor:"pointer", fontSize:12, fontFamily:"inherit", letterSpacing:1, fontWeight:700, transition:"background .2s" },
  approveBtn:{ flex:1, padding:"10px", borderRadius:8, border:"1px solid rgba(0,255,136,.4)", background:"rgba(0,255,136,.07)", color:"#00ff88", cursor:"pointer", fontSize:12, fontFamily:"inherit", letterSpacing:1, fontWeight:700, transition:"background .2s" },

  /* Toast */
  toast:    { position:"fixed", bottom:24, right:24, background:"#000d22", border:"1px solid rgba(0,212,255,.35)", borderRadius:14, padding:"14px 16px", maxWidth:300, zIndex:50, boxShadow:"0 0 40px rgba(0,212,255,.12)", animation:"toastIn .4s ease" },
  toastHeader:{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:8 },
  toastOrb: { width:8, height:8, borderRadius:"50%", background:"#00d4ff", boxShadow:"0 0 8px #00d4ff", animation:"blink 1s step-end infinite" },
  toastClose:{ background:"none", border:"none", color:"rgba(0,212,255,.5)", cursor:"pointer", fontSize:18, lineHeight:1, padding:0, fontFamily:"inherit" },
  toastMsg: { color:"#a0d4ff", fontSize:12.5, lineHeight:1.5 },
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CSS ANIMATIONS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const CSS = `
  * { box-sizing: border-box; }
  body { margin: 0; background: #000813; }
  ::-webkit-scrollbar { width: 3px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(0,212,255,.25); border-radius: 3px; }
  textarea:focus { border-color: rgba(0,212,255,.5) !important; box-shadow: 0 0 14px rgba(0,212,255,.08); }
  textarea::placeholder { color: rgba(0,212,255,.28); }

  @keyframes spin    { to { transform: translate(-50%,-50%) rotate(360deg); } }
  @keyframes scan    { 0% { top:-2px; } 100% { top:100vh; } }
  @keyframes blink   { 0%,100%{opacity:1;} 50%{opacity:0;} }
  @keyframes fadeUp  { from{opacity:0;transform:translateY(6px);} to{opacity:1;transform:none;} }
  @keyframes bounce  { 0%,60%,100%{transform:translateY(0);} 30%{transform:translateY(-6px);} }
  @keyframes pulse   { 0%{transform:translate(-50%,-50%) scale(1);opacity:.8;} 100%{transform:translate(-50%,-50%) scale(1.7);opacity:0;} }
  @keyframes toastIn { from{opacity:0;transform:translateY(12px);} to{opacity:1;transform:none;} }
`;

