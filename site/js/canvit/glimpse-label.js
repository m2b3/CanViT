// How a glimpse is labeled: the paper's index t, from 0, by default. A page whose charts count glimpses from 1, as a
// talk's do, sets data-glimpse-label="count" on <html>, and its glimpses read "Glimpse 1", "Glimpse 2", and so on, as
// the talk's captions name them.
export function glimpseLabel(t) {
  return document.documentElement.dataset.glimpseLabel === "count" ? `Glimpse ${t + 1}` : `t = ${t}`;
}
