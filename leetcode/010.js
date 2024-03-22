var isMatch = function (s, p) {
  let sIdx = 0;
  let lastChar = "";
  let starBreakChar = "";

  for (let pIdx = 0; pIdx < p.length; pIdx++) {
    const pChar = p[pIdx];

    switch (pChar) {
      case "*": {
        console.log(`repeat command! Repeat: ${lastChar}`);
        if (lastChar === ".") {
          if (p[pIdx + 1] != null) {
            starBreakChar = p[pIdx + 1];
            console.log(`breaking char: ${starBreakChar}`);
          } else {
            return true;
          }
        }

        while (s[sIdx] === lastChar || (lastChar === "." && s[sIdx] !== starBreakChar)) {
          if (!s[sIdx]) return false;
          sIdx += 1;
        }

        starBreakChar = "";
        lastChar = "";

        break;
      }

      default: {
        console.log(`check letter ${s[sIdx]}`);

        // char could be repeated 0 times with * char
        if (p[pIdx + 1] === "*") {
          lastChar = pChar;
          console.log(sIdx, s.length - sIdx - 1);
          continue;
        }

        if (pChar === s[sIdx] || pChar === ".") {
          console.log(`${pChar} and ${s[sIdx]} are same!`);
          sIdx++;
        } else {
          return false;
        }
      }
    }
  }

  if (sIdx !== s.length) return false;

  return true;
};

console.log(isMatch("aaa", "a*a"));
