// Chapter navigation keeps the existing simulations mounted and their state intact.
export function initPaperDeck(onChange=()=>{}){
 const pages=[...document.querySelectorAll('[data-paper-page]')];
 const stage=document.querySelector('#paper-deck');
 const navigation=document.querySelector('#page-navigation');
 const next=document.querySelector('#page-next');
 const nextLabel=document.querySelector('#page-next-label');
 const nextArrow=document.querySelector('#page-next-arrow');
 const announcement=document.querySelector('#page-announcement');
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 let current=-1,transition=0;

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
  // Cancel any interrupted transition before establishing the next pair of cards.
  pages.forEach(page=>{
   page.getAnimations().forEach(animation=>animation.cancel());
   const active=page===to;
   page.hidden=page!==from&&!active;
   page.inert=!active;
   page.setAttribute('aria-hidden',String(!active));
  });
  current=index;
  document.body.dataset.activePage=to.id;
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
