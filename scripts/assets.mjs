import fs from 'node:fs/promises';
import QRCode from 'qrcode';
await fs.mkdir('assets/katex/fonts',{recursive:true});
for (const name of ['katex.min.js','katex.min.css']) await fs.copyFile(`node_modules/katex/dist/${name}`,`assets/katex/${name}`);
for (const file of await fs.readdir('node_modules/katex/dist/fonts')) if(file.endsWith('.woff2'))await fs.copyFile(`node_modules/katex/dist/fonts/${file}`,`assets/katex/fonts/${file}`);
// Strip fallback formats: every supported browser loads WOFF2.
let css=await fs.readFile('assets/katex/katex.min.css','utf8');
css=css.replace(/,url\([^)]*\.woff\) format\("woff"\),url\([^)]*\.ttf\) format\("truetype"\)/g,'');
await fs.writeFile('assets/katex/katex.min.css',css);
await fs.copyFile('node_modules/katex/LICENSE','assets/katex/LICENSE');
const url='https://muyao-kong.github.io/scenario-actor/';
await fs.writeFile('assets/scenario-actor-qr.svg',await QRCode.toString(url,{type:'svg',errorCorrectionLevel:'H',margin:4,color:{dark:'#292c26',light:'#ffffff'}}));
await QRCode.toFile('assets/scenario-actor-qr.png',url,{width:1600,errorCorrectionLevel:'H',margin:4,color:{dark:'#292c26',light:'#ffffff'}});
console.log('Local math assets and permanent-URL QR code generated.');
