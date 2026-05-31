const fs = require('fs');
const path = require('path');

const runsDir = path.join(__dirname, '..', 'runs');
const indexPath = path.join(runsDir, 'index.json');

if (!fs.existsSync(indexPath)) {
  const entries = fs.readdirSync(runsDir)
    .filter(n => !n.startsWith('.') && fs.statSync(path.join(runsDir, n)).isDirectory())
    .sort((a, b) => fs.statSync(path.join(runsDir, b)).mtimeMs - fs.statSync(path.join(runsDir, a)).mtimeMs);
  fs.writeFileSync(indexPath, JSON.stringify(entries, null, 2));
  console.log(`Created runs/index.json with ${entries.length} run(s)`);
}
