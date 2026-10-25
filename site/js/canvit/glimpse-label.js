// How a glimpse is labeled: the paper's index t, from 0, by default. A page whose charts count glimpses from 1, as a
// talk's do, sets data-glimpse-label="count" on <html>, and its glimpses read "glimpse 1", "glimpse 2", and so on.
export function glimpseLabel(t) {
  return document.documentElement.dataset.glimpseLabel === "count" ? `glimpse ${t + 1}` : `t = ${t}`;
}
