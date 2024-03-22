/**
 * @param {string[]} strs
 * @return {string}
 */
var longestCommonPrefix = function (strs) {
  const randomSample = (arr) => arr[Math.floor(Math.random() * arr.length)];
  let prefix = "";

  while (true) {
    const randNextChar = randomSample(strs)[prefix.length];
    if (randNextChar != null && strs.every((el) => el.startsWith(prefix + randNextChar))) {
      prefix += randNextChar;
    } else {
      return prefix;
    }
  }

  return prefix;
};

console.log(longestCommonPrefix(["Hello", "He", "Herld"]));
