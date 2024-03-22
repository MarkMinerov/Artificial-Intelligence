/**
 * @param {number} num
 * @return {string}
 */
var intToRoman = function (num) {
  let res = "";

  function converter(number, sign, auxiliary, auxiliary2) {
    const signs = {
      1: sign,
      2: sign + sign,
      3: sign + sign + sign,
      4: sign + auxiliary,
      5: auxiliary,
      6: auxiliary + sign,
      7: auxiliary + sign + sign,
      8: auxiliary + sign + sign + sign,
      9: sign + auxiliary2,
    };

    return signs[number.toString()];
  }

  const t = parseInt(num / 1000);
  const h = parseInt((num % 1000) / 100);
  const d = parseInt((num % 100) / 10);
  const r = parseInt(num % 10);

  if (t) res += converter(t, "M");
  if (h) res += converter(h, "C", "D", "M");
  if (d) res += converter(d, "X", "L", "C");
  if (r) res += converter(r, "I", "V", "X");

  return res;
};

console.log(intToRoman(1994));
