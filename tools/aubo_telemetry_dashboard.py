#!/usr/bin/env python3
"""浏览器显示 AUBO HybridIL 预测/实测遥测，不直接控制硬件。"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>AUBO HybridIL 实时监视</title>
<style>
body{background:#10151d;color:#e7edf7;font:14px system-ui;margin:20px}h1{margin:0 0 8px}
#status{color:#9fb3c8;margin-bottom:14px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.card{background:#18212d;border:1px solid #304257;border-radius:8px;padding:10px}canvas{width:100%;height:290px;background:#0c1118}
.legend{font-size:12px;color:#aebed1;margin:5px 0}.warn{color:#ffbf69}.controls{margin:12px 0;padding:10px;background:#18212d;border:1px solid #304257;border-radius:8px}.controls button{margin-left:8px;padding:6px 10px}.stop{background:#9d1c28;color:white}.reset{background:#8a5a12;color:white}
</style></head><body>
<h1>AUBO HybridIL：预测与实测</h1>
<div id="status">等待遥测文件…</div>
<div class="controls"><b>本机部署控制</b>　最大推理次数 <input id="calls" type="number" min="1" value="10" style="width:65px"><button onclick="sendCmd('set_policy_calls')">更新次数</button><button class="stop" onclick="sendCmd('software_stop')">软件停止</button><button class="reset" onclick="sendCmd('reset_init_pos')">复位到 init_pos</button><span id="commandStatus"></span></div>
<div class="grid">
<div class="card"><b>TCP 位置（AUBO_BASE，m）</b><div class="legend">实线：实际；虚线：模型计划</div><canvas id="pos"></canvas></div>
<div class="card"><b>TCP 姿态（AUBO_BASE，rad）</b><div class="legend">实线：实际；虚线：模型计划</div><canvas id="rot"></canvas></div>
<div class="card"><b>三维力（TCP，N）</b><div class="legend">实线：XJC实测；虚线：模型目标</div><canvas id="force"></canvas></div>
<div class="card"><b>三维力矩（TCP，Nm）</b><div class="legend">实线：XJC实测；虚线：模型目标</div><canvas id="torque"></canvas></div>
</div>
<p class="warn">“软件停止”不等于实体急停；真正紧急情况必须使用控制柜/示教器实体急停。复位将停止轨迹、低速MoveJ到配置的 init_pos，并结束本次部署。</p>
<script>
const colors=['#ff6b6b','#4dabf7','#51cf66']; let windowSeconds=60;
function quatRPY(q){if(!q)return null;let[x,y,z,w]=q;let sinr=2*(w*x+y*z),cosr=1-2*(x*x+y*y);let roll=Math.atan2(sinr,cosr);let sinp=2*(w*y-z*x),pitch=Math.abs(sinp)>=1?Math.sign(sinp)*Math.PI/2:Math.asin(sinp);let siny=2*(w*z+x*y),cosy=1-2*(y*y+z*z);return[roll,pitch,Math.atan2(siny,cosy)]}
function val(event,key,rot){let v=event[key];return rot?quatRPY(v):v}
function draw(id,actual,planned,labels){let c=document.getElementById(id),ctx=c.getContext('2d'),w=c.width=c.clientWidth*devicePixelRatio,h=c.height=c.clientHeight*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);w=c.clientWidth;h=c.clientHeight;ctx.clearRect(0,0,w,h);let all=[...actual,...planned].flatMap(p=>p.v||[]).filter(Number.isFinite);if(!all.length){ctx.fillStyle='#9fb3c8';ctx.fillText('暂无数据',12,22);return}let t0=Math.min(...actual.concat(planned).map(p=>p.t)),t1=Math.max(...actual.concat(planned).map(p=>p.t)),range=Math.max(1e-6,t1-t0);let lo=Math.min(...all),hi=Math.max(...all);let pad=Math.max((hi-lo)*.12,1e-4);lo-=pad;hi+=pad;ctx.strokeStyle='#304257';ctx.strokeRect(42,10,w-52,h-35);ctx.fillStyle='#aebed1';ctx.font='11px system-ui';ctx.fillText(lo.toExponential(2),2,h-27);ctx.fillText(hi.toExponential(2),2,18);ctx.fillText('0s',42,h-8);ctx.fillText(range.toFixed(1)+'s',w-45,h-8);
function series(data,k,dash){ctx.setLineDash(dash);ctx.strokeStyle=colors[k];ctx.lineWidth=1.5;ctx.beginPath();let began=false;for(const p of data){let y=p.v&&p.v[k];if(!Number.isFinite(y)){began=false;continue}let x=42+(p.t-t0)/range*(w-52),yy=10+(hi-y)/(hi-lo)*(h-45);if(!began){ctx.moveTo(x,yy);began=true}else ctx.lineTo(x,yy)}ctx.stroke()}for(let k=0;k<3;k++){series(actual,k,[]);series(planned,k,[5,4]);ctx.fillStyle=colors[k];ctx.fillText(labels[k],w-38,20+k*13)}ctx.setLineDash([])}
function render(rows){let actual=rows.filter(r=>r.event==='control_sample').map(r=>({t:r.timestamp_unix_s,v:r.actual_tcp_xyz_aubo_base,force:r.measured_wrench_tcp,rot:quatRPY(r.actual_tcp_quat_xyzw_aubo_base)}));let plan=rows.filter(r=>r.event==='planned_action').map(r=>({t:r.scheduled_timestamp_unix_s||r.timestamp_unix_s,v:r.tcp_xyz_aubo_base,force:r.predicted_wrench_tcp,rot:quatRPY(r.tcp_quat_xyzw_aubo_base)}));let end=Math.max(...actual.concat(plan).map(x=>x.t),0),begin=end-windowSeconds;actual=actual.filter(x=>x.t>=begin);plan=plan.filter(x=>x.t>=begin);draw('pos',actual,plan,['X','Y','Z']);draw('rot',actual.map(x=>({t:x.t,v:x.rot})),plan.map(x=>({t:x.t,v:x.rot})),['Rx','Ry','Rz']);draw('force',actual.map(x=>({t:x.t,v:x.force&&x.force.slice(0,3)})),plan.map(x=>({t:x.t,v:x.force&&x.force.slice(0,3)})),['Fx','Fy','Fz']);draw('torque',actual.map(x=>({t:x.t,v:x.force&&x.force.slice(3,6)})),plan.map(x=>({t:x.t,v:x.force&&x.force.slice(3,6)})),['Tx','Ty','Tz']);let calls=new Set(rows.filter(r=>r.event==='raw_prediction').map(r=>r.policy_call));document.getElementById('status').textContent=`已读取 ${rows.length} 条记录；策略调用 ${calls.size} 次；最新计划 ${plan.length} 步；显示最近 ${windowSeconds} 秒`}
async function tick(){try{let r=await fetch('/api/telemetry?'+Date.now());let d=await r.json();render(d.records)}catch(e){document.getElementById('status').textContent='读取失败：'+e}}tick();setInterval(tick,500);
async function sendCmd(command){let text=command==='software_stop'?'确认软件停止当前部署？这不替代实体急停。':command==='reset_init_pos'?'确认停止当前轨迹并低速复位到配置的 init_pos？复位后本次部署将结束。':'更新本次部署允许的最大推理次数？';if(!confirm(text))return;let value=command==='set_policy_calls'?Number(document.getElementById('calls').value):null;try{let r=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command,value})});let d=await r.json();document.getElementById('commandStatus').textContent=d.ok?'命令已写入本机部署队列':'命令失败：'+d.error}catch(e){document.getElementById('commandStatus').textContent='发送失败：'+e}}
</script></body></html>"""


def read_records(path: Path, maximum: int):
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # 部署线程刚写入半行时，下一次轮询再读取即可。
                continue
    return records[-maximum:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", default="logs/aubo_telemetry.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--maximum-records", type=int, default=20000)
    parser.add_argument("--allow-control", action="store_true", help="允许网页写入本机控制命令；部署配置也必须 browser_control.enabled=true")
    parser.add_argument("--command-path", default="logs/aubo_dashboard_commands.jsonl")
    args = parser.parse_args()
    path = Path(args.telemetry)
    command_path = Path(args.command_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            return

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/api/telemetry":
                body = json.dumps({"records": read_records(path, args.maximum_records)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
            elif route in ("/", "/index.html"):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            else:
                self.send_error(404)
                return
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if urlparse(self.path).path != "/api/command":
                self.send_error(404)
                return
            if not args.allow_control:
                body = json.dumps({"ok": False, "error": "dashboard was started without --allow-control"}).encode()
                self.send_response(403)
            else:
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(size))
                    command = payload.get("command")
                    if command not in ("software_stop", "reset_init_pos", "set_policy_calls"):
                        raise ValueError("unsupported command")
                    if command == "set_policy_calls" and int(payload.get("value")) <= 0:
                        raise ValueError("policy call count must be positive")
                    command_path.parent.mkdir(parents=True, exist_ok=True)
                    with command_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"command": command, "value": payload.get("value")}) + "\n")
                    body = json.dumps({"ok": True}).encode()
                    self.send_response(200)
                except Exception as error:
                    body = json.dumps({"ok": False, "error": str(error)}).encode()
                    self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("浏览器监视器：http://{}:{}".format(args.host, args.port))
    print("遥测文件：{}".format(path))
    print("网页控制：{}".format("已启用本机命令写入" if args.allow_control else "未启用（只读）"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
