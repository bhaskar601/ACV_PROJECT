// Capture actual browser views of saved experiment evidence.
const path = require('node:path');
const fs = require('node:fs/promises');
async function main() {
  const pw = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
  const root = path.resolve(__dirname, '..');
  const browser = await pw.chromium.launch({executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true, args:['--disable-gpu']});
  const page = await browser.newPage({viewport:{width:1180,height:650},deviceScaleFactor:2});
  await fs.mkdir(path.join(root,'artifacts/screenshots'),{recursive:true});
  for (const name of ['metrics','execution']) {
    await page.goto('file://'+path.join(root,'artifacts/evidence',name+'.html'));
    await page.screenshot({path:path.join(root,'artifacts/screenshots',name+'.png'),fullPage:true});
  }
  await browser.close();
}
main().catch(e=>{console.error(e);process.exit(1)});
