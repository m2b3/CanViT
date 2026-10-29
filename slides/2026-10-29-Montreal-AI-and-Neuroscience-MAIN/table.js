// The table slide's passive answers: DINOv3's map of each glimpse alone (prob_dinov3.png, both pasted in place), shown
// one glimpse at a time by clipping it to that glimpse's box from the plot step's boxes.json (fractions of the scene),
// as CSS variables that talk.css turns into the clip and the flight from the photograph to the map.

export async function placeTableAnswers(section, boxesUrl) {
  const response = await fetch(boxesUrl);
  if (!response.ok) throw new Error(`table: ${boxesUrl} answered ${response.status}`);
  const boxes = await response.json();
  const answers = section.querySelectorAll(".answer");
  if (answers.length !== boxes.length) throw new Error(`table: ${answers.length} answers for ${boxes.length} glimpses`);
  answers.forEach((answer, i) => {
    for (const key of ["top", "left", "size"]) answer.style.setProperty(`--${key}`, boxes[i][key]);
  });
}
