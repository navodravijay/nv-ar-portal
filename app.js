/* NV. AR Portal Web V1 · Original browser implementation
 * Camera frames remain in this page. No recording or upload code.
 * MediaPipe runtime/model are loaded on Start Camera from pinned/official URLs.
 */
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const RUNTIME_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35';
const EFFECTS = [
  {name:'CYBER',filter:'contrast(1.15) saturate(1.45) hue-rotate(9deg)',accent:'#4de0ff'},
  {name:'RGB SHIFT',filter:'contrast(1.36) saturate(2)',accent:'#d5a3ff'},
  {name:'NEON',filter:'grayscale(1) contrast(2.1) brightness(.8)',accent:'#49ffd7'},
  {name:'AURORA',filter:'hue-rotate(105deg) saturate(2.15) contrast(1.3)',accent:'#94a5ff'},
  {name:'SPECTRAL',filter:'invert(1) hue-rotate(190deg) contrast(1.26)',accent:'#ff96b6'},
  {name:'NOIR',filter:'grayscale(1) contrast(1.85) brightness(1.1)',accent:'#f3f5ff'},
];
const $ = id => document.getElementById(id);
const stage=$('stage'), canvas=$('scene'), ctx=canvas.getContext('2d',{alpha:false}), video=$('camera');
const startBtn=$('startBtn'), stopBtn=$('stopBtn'), cycleBtn=$('cycleBtn'), ecoBtn=$('ecoBtn');
const statusText=$('statusText'), statusLight=$('statusLight'), prompt=$('centerPrompt');
const promptTitle=$('promptTitle'), promptDesc=$('promptDesc');
let landmarker=null, stream=null, running=false, loading=false, raf=0, eco=false, effectIndex=0;
let lastDetect=0, lastVideoTime=-1, lastLandmarks=[], fps=0, frames=0, fpsMark=0;
let smoothCorners=null, lastPortalAt=0, pinchStart=0, pinchLocked=false, lastPinchSwitch=0;
let W=1,H=1;
const reducedMotion=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const effectButtons=Array.from(document.querySelectorAll('.effect'));

function updateStatus(message,live=false){
  if(statusText.textContent!==message) statusText.textContent=message;
  statusLight.classList.toggle('live',live);
}
function showPrompt(title,desc){prompt.classList.remove('hidden');promptTitle.textContent=title;promptDesc.textContent=desc;}
function hidePrompt(){prompt.classList.add('hidden');}
function selectEffect(index){
  effectIndex=((index%EFFECTS.length)+EFFECTS.length)%EFFECTS.length;
  effectButtons.forEach((button,i)=>{
    const selected=i===effectIndex;
    button.classList.toggle('active',selected);
    button.setAttribute('aria-pressed',String(selected));
  });
}
function resize(){
  const r=stage.getBoundingClientRect();
  W=Math.max(1,r.width); H=Math.max(1,r.height);
  const dpr=Math.min(window.devicePixelRatio||1,2);
  canvas.width=Math.round(W*dpr); canvas.height=Math.round(H*dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
}
function pointDist(a,b){return Math.hypot(a.x-b.x,a.y-b.y);}
function pointOnScreen(landmark){
  const vW=video.videoWidth||640, vH=video.videoHeight||480;
  const s=Math.max(W/vW,H/vH), dw=vW*s, dh=vH*s, ox=(W-dw)/2, oy=(H-dh)/2;
  return {x:W-(ox+landmark.x*dw),y:oy+landmark.y*dh};
}
function videoCover(filter='none',dx=0,dy=0){
  const vW=video.videoWidth||640,vH=video.videoHeight||480;
  const s=Math.max(W/vW,H/vH),dw=vW*s,dh=vH*s,ox=(W-dw)/2,oy=(H-dh)/2;
  ctx.save();ctx.translate(W,0);ctx.scale(-1,1);ctx.filter=filter;
  ctx.drawImage(video,ox-dx,oy+dy,dw,dh);
  ctx.restore();
}
function drawBackground(now){
  ctx.fillStyle='#071120';ctx.fillRect(0,0,W,H);
  const tt=now*.0003;
  const glow=ctx.createRadialGradient(W*.50,H*.53,0,W*.50,H*.53,W*.58);
  glow.addColorStop(0,'#15335e');glow.addColorStop(.42,'#0b2040');glow.addColorStop(1,'#050a14');
  ctx.fillStyle=glow;ctx.fillRect(0,0,W,H);
  ctx.save();ctx.strokeStyle='rgba(77,141,255,.095)';ctx.lineWidth=1;
  const grid=36;
  for(let x=(W/2)%grid;x<W;x+=grid){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
  for(let y=(H/2)%grid;y<H;y+=grid){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
  const cx=W/2,cy=H/2,r=Math.min(W,H)*.27;
  for(let ring=0;ring<3;ring++){
    ctx.save();ctx.translate(cx,cy);ctx.rotate(tt*(ring%2?-.8:1));
    ctx.beginPath();ctx.arc(0,0,r+ring*30,-Math.PI*.73,Math.PI*.44+ring*.08);
    ctx.strokeStyle=`rgba(77,224,255,${.19-ring*.043})`;ctx.lineWidth=ring===0?2:1;ctx.stroke();ctx.restore();
  }
  for(let i=0;i<30;i++){
    const a=i*2.399+tt*(.21+(i%3)*.04),rad=r*(1.18+(i%5)*.14);
    const x=cx+Math.cos(a)*rad,y=cy+Math.sin(a)*rad*.69;
    ctx.fillStyle=i%6===0?'#6adfff':'#5f9ce062';ctx.fillRect(x,y,i%4===0?2:1,i%4===0?2:1);
  }
  ctx.restore();
}
function drawCameraBackground(){
  videoCover('contrast(1.04) saturate(.7) brightness(.7)');
  ctx.fillStyle='rgba(5,12,28,.45)';ctx.fillRect(0,0,W,H);
  const gradient=ctx.createLinearGradient(0,0,W,H);gradient.addColorStop(0,'#0923426b');gradient.addColorStop(.5,'#0a102800');gradient.addColorStop(1,'#101a3f91');
  ctx.fillStyle=gradient;ctx.fillRect(0,0,W,H);
}
function pathPortal(corners){
  ctx.beginPath();ctx.moveTo(corners[0].x,corners[0].y);
  for(let i=1;i<corners.length;i++)ctx.lineTo(corners[i].x,corners[i].y);
  ctx.closePath();
}
function centroid(points){return {x:points.reduce((v,p)=>v+p.x,0)/points.length,y:points.reduce((v,p)=>v+p.y,0)/points.length};}
function portalFromHands(hands){
  if(hands.length<2)return null;
  const pairs=hands.slice(0,2).map(hand=>[pointOnScreen(hand[4]),pointOnScreen(hand[8])]);
  pairs.sort((a,b)=>(a[0].x+a[1].x)-(b[0].x+b[1].x));
  const left=pairs[0].sort((a,b)=>a.y-b.y),right=pairs[1].sort((a,b)=>a.y-b.y);
  const corners=[left[0],right[0],right[1],left[1]];
  const midLeft=centroid(left),midRight=centroid(right);
  const width=pointDist(midLeft,midRight),height=(pointDist(...left)+pointDist(...right))/2;
  if(width<Math.min(80,W*.2)||height<Math.min(32,H*.08))return null;
  return corners;
}
function lerpCorners(next,dt){
  if(!smoothCorners){smoothCorners=next.map(p=>({...p}));return smoothCorners;}
  const blend=1-Math.exp(-Math.min(dt,100)*.022);
  for(let i=0;i<4;i++){
    smoothCorners[i].x+=(next[i].x-smoothCorners[i].x)*blend;
    smoothCorners[i].y+=(next[i].y-smoothCorners[i].y)*blend;
  }
  return smoothCorners;
}
function drawGrid(corners,now){
  const lerp=(a,b,t)=>({x:a.x+(b.x-a.x)*t,y:a.y+(b.y-a.y)*t});
  ctx.save();ctx.strokeStyle='#a5eeff31';ctx.lineWidth=.75;
  for(let u=.12;u<1;u+=.12){
    const a=lerp(corners[0],corners[3],u),b=lerp(corners[1],corners[2],u);
    ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
  }
  for(let u=.1;u<1;u+=.1){
    const a=lerp(corners[0],corners[1],u),b=lerp(corners[3],corners[2],u);
    ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
  }
  const beam=((now*.00021)%1),a=lerp(corners[0],corners[3],beam),b=lerp(corners[1],corners[2],beam);
  ctx.strokeStyle='#91f8ff8b';ctx.lineWidth=1.6;ctx.shadowColor='#4de0ff';ctx.shadowBlur=14;
  ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();ctx.restore();
}
function drawPortal(corners,now){
  const effect=EFFECTS[effectIndex];const c=centroid(corners);
  ctx.save();pathPortal(corners);ctx.clip();
  videoCover(effect.filter);
  if(effectIndex===1){
    ctx.save();ctx.globalCompositeOperation='screen';ctx.globalAlpha=.29;
    videoCover('sepia(1) saturate(7) hue-rotate(295deg) contrast(1.5)',-6);
    videoCover('sepia(1) saturate(7) hue-rotate(135deg) contrast(1.5)',6);
    ctx.restore();
  }
  if(effectIndex===2){ctx.fillStyle='rgba(0,251,225,.17)';ctx.fillRect(0,0,W,H);}
  if(effectIndex===3){
    const aurora=ctx.createLinearGradient(0,0,W,H);aurora.addColorStop(0,'#4a3eff66');aurora.addColorStop(.52,'#13dbdb11');aurora.addColorStop(1,'#ff57b555');ctx.fillStyle=aurora;ctx.fillRect(0,0,W,H);
  }
  if(effectIndex===0||effectIndex===3)drawGrid(corners,now);
  if(effectIndex===5){ctx.fillStyle='#03071932';ctx.fillRect(0,0,W,H);}
  ctx.fillStyle='rgba(3,15,29,.12)';for(let y=0;y<H;y+=5)ctx.fillRect(0,y,W,1);
  ctx.restore();
  ctx.save();ctx.shadowColor=effect.accent;ctx.shadowBlur=eco?10:23;pathPortal(corners);ctx.strokeStyle=effect.accent;ctx.lineWidth=2.3;ctx.stroke();ctx.restore();
  ctx.save();pathPortal(corners);ctx.strokeStyle='#f7ffff88';ctx.lineWidth=.75;ctx.stroke();
  corners.forEach((p,i)=>{
    ctx.strokeStyle=i%2?'#c4e5ff':'#4de0ff';ctx.lineWidth=2.4;
    ctx.beginPath();ctx.arc(p.x,p.y,5,0,Math.PI*2);ctx.stroke();
    ctx.fillStyle='#d7fbff';ctx.beginPath();ctx.arc(p.x,p.y,1.7,0,Math.PI*2);ctx.fill();
  });
  ctx.textAlign='center';ctx.font='700 10px ui-monospace,monospace';ctx.fillStyle='#c8faff';
  const upper=(corners[0].y+corners[1].y)/2;
  ctx.fillText('NV. // '+effect.name,c.x,Math.max(15,upper-14));
  if(!eco){
    for(let i=0;i<14;i++){
      const a=i*2.399+now*.00024,rr=24+(i*27%110);
      const x=c.x+Math.cos(a)*rr,y=c.y+Math.sin(a)*rr*.55;
      ctx.fillStyle=i%4===0?'#d1fbff':'#4de0ff88';ctx.fillRect(x,y,1.6,1.6);
    }
  }
  ctx.restore();
}
function render(now,dt){
  if(!running){drawBackground(now);return;}
  if(video.readyState<2){drawBackground(now);return;}
  drawCameraBackground();
  const quad=portalFromHands(lastLandmarks);
  if(quad){lastPortalAt=now;drawPortal(lerpCorners(quad,dt),now);}
  else{
    smoothCorners=null;
    ctx.save();ctx.strokeStyle='#7bf0ff5b';ctx.lineWidth=1;ctx.setLineDash([5,11]);
    ctx.strokeRect(W*.32,H*.28,W*.36,H*.43);ctx.setLineDash([]);ctx.restore();
  }
}
function handlePinch(hands,now){
  if(hands.length!==1){pinchStart=0;pinchLocked=false;return;}
  const hand=hands[0],thumb=hand[4],index=hand[8];
  const palm=pointDist(hand[5],hand[17]);
  if(palm<=.0001)return;
  const ratio=pointDist(thumb,index)/palm;
  if(ratio<.38){
    if(!pinchStart)pinchStart=now;
    if(!pinchLocked && now-pinchStart>=700 && now-lastPinchSwitch>1200){
      selectEffect(effectIndex+1);pinchLocked=true;lastPinchSwitch=now;
      updateStatus('EFFECT CHANGED · '+EFFECTS[effectIndex].name,true);
    }
  }else if(ratio>.58){pinchStart=0;pinchLocked=false;}
}
function updateHandUI(count){
  const n=count||0;$('handLabel').textContent=n+' HAND'+(n===1?'':'S');
  if(n===2){hidePrompt();updateStatus('PORTAL TRACKING · TWO HANDS DETECTED',true);}
  else if(n===1){showPrompt('ONE MORE HAND','Raise both hands to build a portal. Pinch to switch effects.');updateStatus('HAND DETECTED · READY FOR PORTAL',true);}
  else{showPrompt('SHOW YOUR HANDS','Bring two hands into camera view to form a portal.');updateStatus('CAMERA ACTIVE · SEARCHING FOR HANDS',true);}
}
function loop(now){
  if(!running)return;
  const delta=now-(loop.previous||now);loop.previous=now;
  if(video.readyState>=2 && now-lastDetect>(eco?70:36) && video.currentTime!==lastVideoTime){
    try{
      const result=landmarker.detectForVideo(video,now);
      lastLandmarks=result.landmarks||[];
      lastVideoTime=video.currentTime;lastDetect=now;
      handlePinch(lastLandmarks,now);updateHandUI(lastLandmarks.length);
      frames++;
      if(now-fpsMark>1000){fps=Math.round(frames*1000/Math.max(1,now-fpsMark));$('fpsLabel').textContent='TRACK '+fps+' FPS';fpsMark=now;frames=0;}
    }catch(error){console.error('Hand tracking failed:',error);stop();showPrompt('TRACKING INTERRUPTED','Try Start Camera again.');updateStatus('TRACKING ERROR · RETRY',false);return;}
  }
  render(now,delta);raf=requestAnimationFrame(loop);
}
async function createLandmarker(){
  if(landmarker)return landmarker;
  const {HandLandmarker,FilesetResolver}=await import(`${RUNTIME_BASE}/vision_bundle.mjs`);
  const wasm=await FilesetResolver.forVisionTasks(`${RUNTIME_BASE}/wasm`);
  const options=delegate=>({baseOptions:{modelAssetPath:MODEL_URL,delegate},runningMode:'VIDEO',numHands:2,minHandDetectionConfidence:.58,minTrackingConfidence:.55,minHandPresenceConfidence:.55});
  try{landmarker=await HandLandmarker.createFromOptions(wasm,options('GPU'));}
  catch(err){console.warn('GPU unavailable; switching to CPU.',err);landmarker=await HandLandmarker.createFromOptions(wasm,options('CPU'));}
  return landmarker;
}
async function start(){
  if(running||loading)return;
  loading=true;startBtn.disabled=true;startBtn.textContent='INITIALIZING ...';
  showPrompt('INITIALIZING AI','Loading MediaPipe runtime and hand tracking model.');updateStatus('LOADING AI MODEL',false);
  try{
    if(!navigator.mediaDevices?.getUserMedia)throw new Error('Camera needs HTTPS or localhost and a supported browser.');
    await createLandmarker();
    updateStatus('REQUESTING CAMERA ACCESS',false);
    stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:'user',width:{ideal:960},height:{ideal:720},frameRate:{ideal:30,max:30}}});
    video.srcObject=stream;
    await video.play();
    if(!video.videoWidth){await new Promise((resolve,reject)=>{const timeout=setTimeout(()=>reject(new Error('Camera did not start.')),8000);video.addEventListener('loadedmetadata',()=>{clearTimeout(timeout);resolve();},{once:true});});}
    running=true;loading=false;loop.previous=0;lastVideoTime=-1;lastDetect=0;lastLandmarks=[];smoothCorners=null;fpsMark=performance.now();frames=0;
    startBtn.textContent='● CAMERA LIVE';stopBtn.disabled=false;cycleBtn.disabled=false;
    showPrompt('SHOW YOUR HANDS','Raise two hands to form your interactive portal.');updateStatus('CAMERA ACTIVE · SEARCHING FOR HANDS',true);
    raf=requestAnimationFrame(loop);
  }catch(err){
    console.error('Cannot start AR Portal:',err);
    stop();
    let detail='Model or camera could not start. Check browser permissions and internet access.';
    if(err.name==='NotAllowedError')detail='Camera permission was denied. Allow access in browser settings and retry.';
    else if(err.name==='NotFoundError')detail='No camera detected. Connect a webcam and retry.';
    else if(err.name==='NotReadableError')detail='Camera is in use by another app. Close it and retry.';
    else if(err.message?.includes('HTTPS'))detail=err.message;
    showPrompt('START FAILED',detail);
    updateStatus('START FAILED · CHECK PERMISSIONS',false);
  }finally{loading=false;startBtn.disabled=false;if(!running)startBtn.textContent='◉ START CAMERA  ↗';}
}
function stop(){
  running=false;cancelAnimationFrame(raf);
  if(stream){stream.getTracks().forEach(track=>track.stop());stream=null;}
  video.pause();video.srcObject=null;lastLandmarks=[];smoothCorners=null;pinchStart=0;pinchLocked=false;
  startBtn.disabled=false;startBtn.textContent='◉ START CAMERA  ↗';stopBtn.disabled=true;cycleBtn.disabled=true;
  $('handLabel').textContent='0 HANDS';$('fpsLabel').textContent='FPS --';
  showPrompt('YOUR PORTAL AWAITS','Enable your camera to begin the experience.');updateStatus('SYSTEM READY · CAMERA OFF',false);
  drawBackground(performance.now());
}
startBtn.addEventListener('click',start);stopBtn.addEventListener('click',stop);
cycleBtn.addEventListener('click',()=>selectEffect(effectIndex+1));
ecoBtn.addEventListener('click',()=>{eco=!eco;ecoBtn.setAttribute('aria-pressed',String(eco));$('ecoState').textContent=eco?'ON':'OFF';});
effectButtons.forEach(button=>button.addEventListener('click',()=>selectEffect(Number(button.dataset.effect))));
document.addEventListener('keydown',event=>{
  if(event.altKey||event.ctrlKey||event.metaKey||event.target instanceof HTMLInputElement)return;
  const key=event.key.toLowerCase();
  if(/^[1-6]$/.test(key))selectEffect(Number(key)-1);
  if(key===' '){event.preventDefault();selectEffect(effectIndex+1);}
  if(key==='e')ecoBtn.click();
  if(key==='q'||key==='escape')stop();
});
window.addEventListener('pagehide',stop);
document.addEventListener('visibilitychange',()=>{if(document.hidden&&running)stop();});
const ro=new ResizeObserver(resize);ro.observe(stage);resize();selectEffect(0);drawBackground(performance.now());
if(!reducedMotion){
  // Ambient animation only when camera is off. Throttled to reduce idle CPU usage.
  let lastIdle=0;
  function idleLoop(now){if(!running&&now-lastIdle>70){drawBackground(now);lastIdle=now;}requestAnimationFrame(idleLoop);}
  requestAnimationFrame(idleLoop);
}
