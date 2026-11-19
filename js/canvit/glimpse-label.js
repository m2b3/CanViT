// A glimpse's label: the paper's index, "t = 0", by default, or "Glimpse 1" on a page that counts glimpses from 1
// (data-glimpse-label="count" on <html>).
export function glimpseLabel(t) {
  return document.documentElement.dataset.glimpseLabel === "count" ? `Glimpse ${t + 1}` : `t = ${t}`;
}
