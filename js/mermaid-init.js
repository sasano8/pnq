(() => {
  const render = () => {
    if (typeof mermaid === "undefined") return;
    const palette = (document.body.getAttribute("data-md-color-scheme") || "").trim();
    mermaid.initialize({
      startOnLoad: false,
      theme: palette === "slate" ? "dark" : "default",
      securityLevel: "loose",
    });
    const nodes = document.querySelectorAll("div.mermaid:not([data-processed])");
    if (nodes.length) mermaid.run({ nodes });
  };

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(render);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", render);
  } else {
    render();
  }
})();
