from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from app.pipeline.orchestrator import GraphRAGv3
from app.api.v1 import router as api_v1_router
import re
import logging

app = FastAPI()

# ── Include API v1 router ─────────────────────────────────────────────────
app.include_router(api_v1_router)

chatbot = GraphRAGv3()

# attach optional middlewares
from app.middleware import APIKeyAuthMiddleware, SimpleRateLimitMiddleware
app.add_middleware(APIKeyAuthMiddleware)
app.add_middleware(SimpleRateLimitMiddleware)

from app.metrics import init_metrics, metrics_response, request_counter
init_metrics()
from app.tracing import init_tracing


@app.on_event("startup")
def _maybe_init_tracing():
  import os
  if os.environ.get('ENABLE_OTEL', 'false').lower() in ('1', 'true', 'yes'):
    try:
      init_tracing(app)
    except Exception:
      pass


@app.on_event("startup")
def check_neo4j_indexes_on_startup():
  # Only run startup checks when explicitly enabled to avoid blocking test environments
  import os
  if os.environ.get('RUN_NEO4J_STARTUP_CHECK', 'false').lower() not in ('1','true','yes'):
    return
  try:
    missing = chatbot.db.check_indexes()
    if missing:
      logging.getLogger(__name__).warning("Missing Neo4j indexes/constraints: %s", missing)
    else:
      logging.getLogger(__name__).info("Neo4j indexes OK")
  except Exception:
    logging.getLogger(__name__).exception("Failed to check Neo4j indexes on startup")


  @app.on_event("startup")
  def check_faiss_on_startup():
    # Record FAISS availability and meta info into app.state for health checks
    try:
      faiss_info = {"available": False, "index_count": 0}
      retriever = getattr(chatbot, 'retriever', None)
      if retriever is not None:
        embed = getattr(retriever, 'embed', None)
        if embed is not None and getattr(embed, 'faiss_index', None) is not None:
          faiss_info['available'] = True
          try:
            faiss_idx = embed.faiss_index
            # read ntotal if attribute exists
            faiss_info['index_count'] = int(getattr(faiss_idx, 'ntotal', 0) or 0)
          except Exception:
            faiss_info['index_count'] = 0
      app.state.faiss_info = faiss_info
      logging.getLogger(__name__).info('FAISS startup info: %s', faiss_info)
    except Exception:
      logging.getLogger(__name__).exception('Failed to determine FAISS status on startup')

def markdown_to_html(text):
    """Convert markdown to HTML for display in chat"""
    if not text:
        return text
    
    # Bold **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    # Italic *text*
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    
    # Bullet points
    lines = text.split('\n')
    in_list = False
    result = []
    for line in lines:
        if line.strip().startswith('- '):
            if not in_list:
                result.append('<ul>')
                in_list = True
            result.append(f'<li>{line.strip()[2:]}</li>')
        elif line.strip().startswith('| '):
            # Markdown table
            result.append(line.replace('|', '</td><td>').replace('<td>', '<td style="padding:8px;border:1px solid #ccc">').replace('</td></td>', '</td>'))
            if 'Thuộc tính' in line or '---|' in line:
                result[-1] = '<table style="width:100%;border-collapse:collapse;margin:10px 0"><tr>' + result[-1].replace('<td>', '<th>', 1) + '</tr></table>' if 'Thuộc tính' in line else '<tr>' + result[-1] + '</tr>'
        else:
            if in_list and line.strip():
                result.append('</ul>')
                in_list = False
            if line.strip():
                result.append(f'<p>{line}</p>')
    
    if in_list:
        result.append('</ul>')
    
    return '\n'.join(result)

CHAT_HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <title>Graph RAG Chatbot - Tư Vấn Lốp Chuyên Nghiệp</title>
  <style>
    * { box-sizing: border-box; }
    body { 
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
      margin: 0; 
      padding: 0; 
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
    }
    .container { 
      max-width: 800px; 
      margin: 20px auto; 
      padding: 24px; 
      background: #ffffff; 
      border-radius: 16px; 
      box-shadow: 0 20px 60px rgba(0,0,0,0.2);
    }
    h1 { 
      margin: 0 0 8px 0;
      color: #2c3e50;
      font-size: 28px;
    }
    .subtitle {
      color: #7f8c8d;
      font-size: 14px;
      margin-bottom: 20px;
    }
    .chat { 
      min-height: 400px; 
      max-height: 500px;
      border: 1px solid #ecf0f1; 
      border-radius: 12px; 
      padding: 20px; 
      background: #f8f9fa; 
      overflow-y: auto;
      margin-bottom: 20px;
    }
    .message { 
      margin-bottom: 16px; 
      animation: slideIn 0.3s ease-in;
    }
    @keyframes slideIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .user-msg {
      text-align: right;
      margin-left: 40px;
    }
    .user-msg .bubble {
      background: #3498db;
      color: white;
      padding: 12px 16px;
      border-radius: 18px;
      border-bottom-right-radius: 4px;
      display: inline-block;
      max-width: 70%;
      word-wrap: break-word;
    }
    .bot-msg {
      text-align: left;
      margin-right: 40px;
    }
    .bot-msg .bubble {
      background: #ecf0f1;
      color: #2c3e50;
      padding: 14px 16px;
      border-radius: 18px;
      border-bottom-left-radius: 4px;
      display: inline-block;
      max-width: 70%;
      word-wrap: break-word;
      line-height: 1.5;
    }
    .bot-msg strong { color: #2980b9; }
    .bot-msg em { color: #e74c3c; }
    .bot-msg table {
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0;
      font-size: 13px;
    }
    .bot-msg table th, .bot-msg table td {
      padding: 8px;
      border: 1px solid #bdc3c7;
      text-align: center;
    }
    .bot-msg table th { background: #3498db; color: white; }
    .bot-msg ul {
      margin: 8px 0;
      padding-left: 20px;
    }
    .bot-msg li {
      margin: 4px 0;
    }
    .input-row { 
      display: flex; 
      gap: 12px; 
      margin-bottom: 16px;
    }
    input[type=text] { 
      flex: 1; 
      padding: 14px 16px; 
      border: 2px solid #ecf0f1; 
      border-radius: 10px;
      font-size: 15px;
      transition: border-color 0.3s;
    }
    input[type=text]:focus {
      outline: none;
      border-color: #3498db;
    }
    button { 
      padding: 14px 24px; 
      border: none; 
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white; 
      border-radius: 10px; 
      cursor: pointer;
      font-weight: 600;
      font-size: 15px;
      transition: transform 0.2s;
    }
    button:hover { 
      transform: translateY(-2px);
      box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
    }
    button:active {
      transform: translateY(0);
    }
    .small { 
      font-size: 13px; 
      color: #7f8c8d; 
    }
    .hint { 
      margin-top: 12px; 
      padding: 12px;
      background: #ecf0f1;
      border-left: 4px solid #3498db;
      border-radius: 4px;
      font-size: 13px;
      color: #34495e;
    }
    .reset-btn {
      background: #95a5a6;
      padding: 10px 16px;
      font-size: 13px;
    }
    .reset-btn:hover {
      background: #7f8c8d;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🏍️ Chatbot Tư Vấn Lốp Chuyên Nghiệp</h1>
    <p class="subtitle">Hỏi về giá, tốc độ, tải trọng, hoặc so sánh các loại lốp</p>
    <div class="chat" id="chat"></div>
    <div class="input-row">
      <input type="text" id="query" placeholder="Nhập câu hỏi... (ví dụ: Giá lốp 120/70-17 bao nhiêu?)" autocomplete="off" />
      <button onclick="sendQuery()">Gửi</button>
    </div>
    <div class="hint">
      <strong>💡 Gợi ý:</strong> Hãy hỏi cụ thể về kích thước lốp (ví dụ: 120/70-17, 2.50-17) để tôi trả lời chính xác. 
      Sau khi hỏi một lốp, bạn có thể hỏi "Mẫu này..." để so sánh hoặc lấy thêm thông tin.
    </div>
    <div class="hint">
      <button onclick="resetContext()" class="reset-btn">🔄 Reset Context (Bắt đầu lại)</button>
    </div>
  </div>

  <script>
    const chatEl = document.getElementById('chat');
    const queryEl = document.getElementById('query');

    function appendMessage(role, html) {
      const div = document.createElement('div');
      div.className = 'message ' + (role === 'user' ? 'user-msg' : 'bot-msg');
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.innerHTML = html;
      div.appendChild(bubble);
      chatEl.appendChild(div);
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    async function sendQuery() {
      const q = queryEl.value.trim();
      if (!q) return;
      appendMessage('user', q);
      queryEl.value = '';
      
      try {
        const response = await fetch(`/query?q=${encodeURIComponent(q)}`);
        const data = await response.json();
        if (data.error) {
          appendMessage('bot', `<span style="color:#e74c3c"><strong>❌ Lỗi:</strong> ${data.error}</span>`);
        } else {
          appendMessage('bot', data.result);
        }
      } catch (e) {
        appendMessage('bot', `<span style="color:#e74c3c"><strong>❌ Lỗi kết nối:</strong> ${e.message}</span>`);
      }
    }

    async function resetContext() {
      try {
        const response = await fetch('/reset', { method: 'POST' });
        const data = await response.json();
        appendMessage('bot', `<strong>✅ ${data.status}</strong><br>Bạn có thể bắt đầu hỏi từ đầu!`);
      } catch (e) {
        appendMessage('bot', `<span style="color:#e74c3c">Lỗi: ${e.message}</span>`);
      }
    }

    queryEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') sendQuery();
    });
    
    // Show welcome message
    window.addEventListener('load', () => {
      appendMessage('bot', '👋 <strong>Chào bạn!</strong> Tôi là chatbot tư vấn lốp chuyên nghiệp. Hỏi tôi về bất kỳ loại lốp nào - giá, tốc độ, tải, hoặc so sánh giữa các loại. <br><br>💬 Hãy bắt đầu bằng cách nói: "Lốp 120/70-17 giá bao nhiêu?" hoặc "So sánh lốp 100/80-14 và 110/80-14"');
    });
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return CHAT_HTML

@app.get("/query")
def query(q: str):
  try:
    if request_counter is not None:
      try:
        request_counter.labels(endpoint='/query').inc()
      except Exception:
        pass
  except Exception:
    pass
  try:
    result = chatbot.run(q)
    # Convert markdown result to HTML so the frontend can render formatted tables
    try:
      html = markdown_to_html(result)
    except Exception:
      html = result
    return {"result": html}
  except Exception as e:
    return {"error": str(e)}


@app.get('/metrics')
def metrics():
  data, content_type = metrics_response()
  if data is None:
    return {"error": "Prometheus client not installed"}
  from fastapi.responses import Response
  return Response(content=data, media_type=content_type)


@app.get('/health')
def health():
  faiss_info = getattr(app.state, 'faiss_info', None)
  if faiss_info is None:
    # compute on-demand
    try:
      faiss_info = {"available": False, "index_count": 0}
      retriever = getattr(chatbot, 'retriever', None)
      if retriever is not None:
        embed = getattr(retriever, 'embed', None)
        if embed is not None and getattr(embed, 'faiss_index', None) is not None:
          faiss_info['available'] = True
          try:
            faiss_idx = embed.faiss_index
            faiss_info['index_count'] = int(getattr(faiss_idx, 'ntotal', 0) or 0)
          except Exception:
            faiss_info['index_count'] = 0
    except Exception:
      faiss_info = {"available": False, "index_count": 0}
  return JSONResponse(content={"status": "ok", "faiss": faiss_info})

@app.post("/reset")
def reset_context():
    chatbot.reset_context()
    try:
      if request_counter is not None:
        try:
          request_counter.labels(endpoint='/reset').inc()
        except Exception:
          pass
    except Exception:
      pass
    return {"status": "Context đã được reset. Bạn có thể bắt đầu cuộc hội thoại mới!"}
