export default function (eleventyConfig) {
  eleventyConfig.addPassthroughCopy("src/assets");
  eleventyConfig.addFilter("sigfigs", (n, digits) => Number(n).toPrecision(digits));

  return {
    dir: {
      input: "src",
      output: "_site",
    },
  };
}
