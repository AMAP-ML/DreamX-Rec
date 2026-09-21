// Requires Node.js and sharp. Run from any directory with sharp on NODE_PATH.
// README viewers may cap images at 500 CSS pixels and strip inline styles.
// Render four equal-width bands so the complete diagram retains its proportions.
const fs = require('node:fs');
const path = require('node:path');
const sharp = require('sharp');
const assets = path.resolve(__dirname, '../assets');
const name = 'dreamx-rec-architecture';
const scale = 4;
const cuts = [0, 302, 640, 830, 1120];
(async () => {
  const source = fs.readFileSync(path.join(assets, `${name}.svg`));
  const full = await sharp(source, { density: 72 * scale }).png().toBuffer();
  await fs.promises.writeFile(path.join(assets, `${name}.png`), full);
  for (let i = 0; i < cuts.length - 1; i++) {
    await sharp(full).extract({ left: 0, top: cuts[i] * scale,
      width: 1680 * scale, height: (cuts[i + 1] - cuts[i]) * scale })
      .png().toFile(path.join(assets, `${name}-${i + 1}.png`));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
