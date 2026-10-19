// A gesture is armed only at the edge where it begins. Reaching an edge while
// reading never turns the page; the reader must make a fresh outward gesture.
export function edgeGesture({top,bottom}){
 let used=false;
 return (x,y,threshold=64)=>{
  if(used||Math.abs(y)<threshold||Math.abs(y)<Math.abs(x)*1.4)return 0;
  const direction=y>0?1:-1;
  if((direction>0&&!bottom)||(direction<0&&!top))return 0;
  used=true;
  return direction;
 };
}

// Chapter navigation keeps the existing simulations mounted and their state intact.
export function initPaperDeck(onChange=()=>{}){
 const pages=[...document.querySelectorAll('[data-paper-page]')];
 const stage=document.querySelector('#paper-deck');
 const navigation=document.querySelector('#page-navigation');
 const previous=document.querySelector('#page-previous');
 const previousLabel=document.querySelector('#page-previous-label');
 const next=document.querySelector('#page-next');
 const nextLabel=document.querySelector('#page-next-label');
 const nextArrow=document.querySelector('#page-next-arrow');
 const announcement=document.querySelector('#page-announcement');
 const mobile=matchMedia('(max-width: 600px), (hover: none) and (pointer: coarse)');
 let swipeLockedUntil=0;
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 const revealSelector='.hero-copy > *, .hero-figure, .section-heading, .lab-top, .plot-toolbar, .plot, .transport, .actout-transport, .axis-contributions, .controls > *, .equation-strip, .legend, .sampling-details, .table-scroll, .findings > p, .paper-copy > *, .qr-block';
 const reveals=new Map();
 let observer,seen=new WeakSet(),revealDirection=1,entryDelay=0;
 let current=-1,transition=0;
 const tapTimers=new WeakMap();
 function tapCue(link){
  if(reduced.matches||!matchMedia('(hover: none), (max-width: 600px)').matches)return;
  clearTimeout(tapTimers.get(link));
  link.classList.remove('tap-cue');
  void link.offsetWidth;
  link.classList.add('tap-cue');
  tapTimers.set(link,setTimeout(()=>link.classList.remove('tap-cue'),860));
 }

 function stopReveals(){
  observer?.disconnect();
  reveals.forEach(animation=>animation.cancel());
  reveals.clear();
 }

 function watchContent(page,direction,animate){
  seen=new WeakSet();
  revealDirection=direction;
  entryDelay=animate?140:0;
  if(reduced.matches||!('IntersectionObserver' in window))return;
  observer=new IntersectionObserver(entries=>{
   let order=0;
   entries.forEach(({target,isIntersecting})=>{
    if(!isIntersecting||seen.has(target)||page!==pages[current]||target.closest('[hidden]'))return;
    seen.add(target);
    const animation=target.animate([
     {opacity:0,transform:`translateY(${revealDirection*40}px)`},
     {opacity:1,transform:'translateY(0)'}
    ],{duration:680,delay:entryDelay+Math.min(order++,4)*50,easing:'cubic-bezier(.14,1,.28,1)',fill:'backwards'});
    reveals.set(target,animation);
    animation.finished.then(()=>reveals.delete(target)).catch(()=>{});
   });
   entryDelay=0;
  },{root:page,threshold:.08});
  page.querySelectorAll(revealSelector).forEach(element=>observer.observe(element));
 }
 pages.forEach(page=>{
  let lastScroll=page.scrollTop;
  page.addEventListener('scroll',()=>{
   const delta=page.scrollTop-lastScroll;
   if(Math.abs(delta)>1)revealDirection=delta>0?1:-1;
   lastScroll=page.scrollTop;
  },{passive:true});
 });
 reduced.addEventListener('change',()=>{if(reduced.matches){stopReveals();pages.forEach(page=>page.getAnimations().forEach(animation=>animation.finish()));}});

 function destination(hash){
  let id;
  try{id=decodeURIComponent(hash.replace(/^#/,''));}catch{return null;}
  if(!id)return {index:0,target:pages[0]};
  const target=document.getElementById(id);
  const page=target?.closest('[data-paper-page]');
  return page?{index:pages.indexOf(page),target}:null;
 }

 function show(index,{animate=true,record=true,focus=true,target=pages[index],entry}={}){
  if(index===current)return;
  const from=pages[current],to=pages[index],direction=index>current?1:-1;
  const sequence=++transition;
  stopReveals();
  // Cancel interrupted transitions before establishing the next pair of pages.
  pages.forEach(page=>{
   page.getAnimations().forEach(animation=>animation.cancel());
   const active=page===to;
   page.hidden=page!==from&&!active;
   page.inert=!active;
   page.setAttribute('aria-hidden',String(!active));
  });
  swipeLockedUntil=from?performance.now()+800:0;
  current=index;
  document.body.dataset.activePage=to.id;
  previous.hidden=index===0;
  if(index>0){
   previous.href=`#${pages[index-1].id}`;
   previousLabel.textContent=pages[index-1].dataset.paperPage;
   previous.setAttribute('aria-label',`Previous page: ${previousLabel.textContent}`);
  }
  const following=pages[index+1]||pages[0];
  next.href=`#${following.id}`;
  nextLabel.textContent=index===pages.length-1?'Back to top':following.dataset.paperPage;
  nextArrow.hidden=index===pages.length-1;
  next.dataset.last=String(index===pages.length-1);
  next.setAttribute('aria-label',index===pages.length-1?'Back to top':`Next page: ${nextLabel.textContent}`);
  announcement.textContent=`${to.dataset.paperPage}, page ${index+1} of ${pages.length}`;
  document.querySelectorAll('.site-header a[href^="#"]').forEach(link=>{
   if(destination(link.hash)?.index===index)link.setAttribute('aria-current','page');
   else link.removeAttribute('aria-current');
  });
  if(record)history.pushState({paperPage:to.id},'',`#${to.id}`);
  onChange({page:to.id,target,previous:from?.id});
  if(entry)to.scrollTop=entry==='bottom'?to.scrollHeight:0;
  if(focus){const heading=to.querySelector('h1,h2');heading.tabIndex=-1;heading.focus({preventScroll:true});}
  watchContent(to,direction,!!from&&animate);

  if(from&&animate&&!reduced.matches){
   const timing={duration:720,easing:'cubic-bezier(.2,1,.32,1)'};
   from.animate([{transform:'translateY(0)',opacity:1},{transform:`translateY(${-direction*108}%)`,opacity:.35}],timing);
   const entering=to.animate([{transform:`translateY(${direction*108}%)`,opacity:.35},{transform:'translateY(0)',opacity:1}],timing);
   entering.finished.then(()=>{if(sequence===transition)from.hidden=true;}).catch(()=>{});
  }else if(from)from.hidden=true;
 }

 document.addEventListener('click',event=>{
  const link=event.target.closest('a[href^="#"]');
  if(!link||event.defaultPrevented||event.button!==0||event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;
  const dest=destination(link.hash);
  if(!dest)return;
  event.preventDefault();
  if(link===next||link===previous)tapCue(link);
  show(dest.index,{target:dest.target,entry:mobile.matches?(dest.index<current&&dest.index!==0?'bottom':'top'):undefined});
 });
 // Keep native page scrolling and chart gestures. Only an outward gesture
 // started at a boundary can request a chapter transition.
 const blocked=target=>target.closest('input,select,textarea,button,a,summary,[contenteditable],.drag-handle');
 function begin(page){
  if(!mobile.matches||page!==pages[current]||performance.now()<swipeLockedUntil)return null;
  return edgeGesture({top:page.scrollTop<=2,bottom:page.scrollHeight-page.clientHeight-page.scrollTop<=2});
 }
 function turn(direction){
  const index=current+direction;
  if(index<0||index>=pages.length)return;
  const page=pages[current];
  if(direction>0?page.scrollHeight-page.clientHeight-page.scrollTop>2:page.scrollTop>2)return;
  show(index,{entry:direction>0?'top':'bottom'});
 }
 pages.forEach(page=>{
  let touch=null,wheel=null,wheelTime=0,wheelX=0,wheelY=0;
  page.addEventListener('touchstart',event=>{
   touch=null;
   if(event.touches.length!==1||blocked(event.target))return;
   const start=event.touches[0],gesture=begin(page);
   if(gesture)touch={x:start.clientX,y:start.clientY,gesture,top:page.scrollTop<=2,bottom:page.scrollHeight-page.clientHeight-page.scrollTop<=2};
  },{passive:true});
  page.addEventListener('touchmove',event=>{
   if(!touch||event.touches.length!==1){touch=null;return;}
   const point=event.touches[0];
   const x=touch.x-point.clientX,y=touch.y-point.clientY;
   const outward=y>0?touch.bottom&&page.scrollHeight-page.clientHeight-page.scrollTop<=2:touch.top&&page.scrollTop<=2;
   if(Math.abs(y)>Math.abs(x)*1.4&&outward&&event.cancelable)event.preventDefault();
   const direction=touch.gesture(x,y);
   if(direction){
    if(event.cancelable)event.preventDefault();
    touch=null;
    turn(direction);
   }
  },{passive:false});
  page.addEventListener('touchend',()=>{touch=null;},{passive:true});
  page.addEventListener('touchcancel',()=>{touch=null;},{passive:true});
  // Wheel support uses the same boundary rule for narrow windows and trackpads.
  // A continuous wheel burst (including momentum) can turn at most one page.
  page.addEventListener('wheel',event=>{
   if(!mobile.matches||page!==pages[current]||event.ctrlKey||blocked(event.target))return;
   const now=performance.now();
   if(now<swipeLockedUntil){event.preventDefault();wheel=null;wheelTime=now;return;}
   if(now-wheelTime>220){wheel=begin(page);wheelX=0;wheelY=0;}
   wheelTime=now;
   const unit=event.deltaMode===1?16:event.deltaMode===2?page.clientHeight:1;
   wheelX+=event.deltaX*unit;wheelY+=event.deltaY*unit;
   const direction=wheel?.(wheelX,wheelY,72);
   if(direction){event.preventDefault();turn(direction);}
  },{passive:false});
 });
 const restore=()=>{const dest=destination(location.hash)||{index:0};show(dest.index,{record:false,target:dest.target});};
 addEventListener('popstate',restore);
 addEventListener('hashchange',restore);
 document.body.classList.add('deck-ready');
 navigation.hidden=false;
 const initial=destination(location.hash)||{index:0};
 show(initial.index,{animate:false,record:false,focus:false,target:initial.target});
 stage.setAttribute('aria-roledescription','chapters');
 return {get activePage(){return pages[current].id;}};
}
