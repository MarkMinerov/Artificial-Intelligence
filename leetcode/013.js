/**
 * @param {string} s
 * @return {number}
 */
var romanToInt = function (s) {
  const signs = [
    [["M", null, null], 1000],
    [["C", "D", "M"], 100],
    [["X", "L", "C"], 10],
    [["I", "V", "X"], 1],
  ];

  const findPattern = (s1, s2, s3, value) => {
    const patters = {
      [s1]: "1",
      [s1 + s1]: "2",
      [s1 + s1 + s1]: "3",
      [s1 + s2]: "4",
      [s2]: "5",
      [s2 + s1]: "6",
      [s2 + s1 + s1]: "7",
      [s2 + s1 + s1 + s1]: "8",
      [s1 + s3]: "9",
    };

    return patters[value];
  };

  let i = 0;
  let sol = 0;

  while (i < s.length) {
    for (const signList of signs) {
      let convertedNumber = null;
      let stopIndex = i + 3;

      while (convertedNumber == null && stopIndex >= 0) {
        convertedNumber = findPattern(...signList[0], s.slice(i, stopIndex));
        if (!convertedNumber) stopIndex--;
      }

      if (convertedNumber) {
        sol += convertedNumber * signList[1];
        i += stopIndex - i;
      }
    }
  }

  return sol;
};

console.log(romanToInt("MCM"));
