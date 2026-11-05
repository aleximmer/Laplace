window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// Example pages come from notebooks, whose markdown cells keep math as raw
// `$...$` text. MathJax only looks inside elements with a process class, and
// not in their children, so tag every element inside those cells.
function tagNotebookMath() {
  document
    .querySelectorAll(".jp-RenderedMarkdown, .jp-RenderedMarkdown *")
    .forEach((el) => el.classList.add("arithmatex"));
}

// This script loads at the end of the page, before MathJax, so tagging now lets
// MathJax's first typeset pass see the cells.
tagNotebookMath();

document$.subscribe(() => {
  tagNotebookMath();

  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
