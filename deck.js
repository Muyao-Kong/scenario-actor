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
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 const revealSelector='.hero-copy > *, .hero-figure, .section-heading, .lab-top, .plot-toolbar, .plot, .transport, .actout-transport, .axis-contributions, .controls > *, .equation-strip, .legend, .sampling-details, .table-scroll, .findings > p, .paper-copy > *, .qr-block';
 const reveals=new Map();
 let observer,seen=new WeakSet(),revealDirection=1,entryDelay=0;
 let current=-1,transition=0;

 function stopReveals(){
  observer?.disconnect();
  reveals.forEach(animation=>animation.cancel());
  reveals.clear();
 }

 function watchContent(page,direction,animate){
  seen=new WeakSet();
  revealDirection=direction;
  entryDelay=animate?120:0;
  if(reduced.matches||!('IntersectionObserver' in window))return;
  observer=new IntersectionObserver(entries=>{
   let order=0;
   entries.forEach(({target,isIntersecting})=>{
    if(!isIntersecting||seen.has(target)||page!==pages[current]||target.closest('[hidden]'))return;
    seen.add(target);
    const animation=target.animate([
     {opacity:0,transform:`translateY(${revealDirection*28}px)`},
     {opacity:1,transform:'translateY(0)'}
    ],{duration:620,delay:entryDelay+Math.min(order++,4)*45,easing:'cubic-bezier(.16,1,.3,1)',fill:'backwards'});
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

 function show(index,{animate=true,record=true,focus=true,target=pages[index]}={}){
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
  next.setAttribute('aria-label',index===pages.length-1?'Back to top':`Next page: ${nextLabel.textContent}`);
  announcement.textContent=`${to.dataset.paperPage}, page ${index+1} of ${pages.length}`;
  document.querySelectorAll('.site-header a[href^="#"]').forEach(link=>{
   if(destination(link.hash)?.index===index)link.setAttribute('aria-current','page');
   else link.removeAttribute('aria-current');
  });
  if(record)history.pushState({paperPage:to.id},'',`#${to.id}`);
  onChange({page:to.id,target,previous:from?.id});
  if(focus){const heading=to.querySelector('h1,h2');heading.tabIndex=-1;heading.focus({preventScroll:true});}
  watchContent(to,direction,!!from&&animate);

  if(from&&animate&&!reduced.matches){
   const timing={duration:680,easing:'cubic-bezier(.22,1,.36,1)'};
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
  show(dest.index,{target:dest.target});
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
