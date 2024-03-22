/**
 * @param {number[]} height
 * @return {number}
 */
var maxArea = function (height) {
  let left = 0;
  let right = height.length - 1;
  let ans = Number.MIN_SAFE_INTEGER;

  while (left < right) {
    let area = 0;

    if (height[left] < height[right]) {
      area = height[left] * (right - left);
      left++;
    } else {
      area = height[right] * (right - left);
      right--;
    }

    ans = Math.max(ans, area);
  }

  return ans;
};

console.log(maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]));
