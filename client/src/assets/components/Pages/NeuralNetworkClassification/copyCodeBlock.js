document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".copy-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const codeEl = btn.parentElement.querySelector("code");
      if (!codeEl) return;

      navigator.clipboard.writeText(codeEl.innerText).then(() => {
        btn.innerText = "✓ Copied";
        setTimeout(() => {
          btn.innerText = ""; // Clear text and show icon again via React
          const icon = document.createElement("i");
          icon.className = "fa fa-copy"; // Optional fallback if using Font Awesome
          btn.appendChild(icon);
        }, 1500);
      });
    });
  });
});
