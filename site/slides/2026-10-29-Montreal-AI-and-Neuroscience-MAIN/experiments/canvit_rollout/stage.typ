// The paper's Figure 1 for #canvit-rollout, built up a glimpse at a time: the composition the paper uses
// (paper/exporter/diagrams/canvit_overview_content.typ), with the glimpses shown up to the input `shown` (0: the scene
// alone), without panel letters, and counting glimpses from 1 as the talk does. The page is the paper's
// (canvit_overview_standalone.typ), so every stage has the paper's layout. build_deck_data.sh compiles it.
#set document(date: none)
#set page(width: 5.5in, height: 6.3cm, margin: (left: 0.035cm, right: 0.07cm, top: 0cm, bottom: 0cm))
#set text(font: ("Helvetica", "Liberation Sans", "Nimbus Sans"), size: 11pt)
#import "../../../../../paper/exporter/diagrams/canvit_overview_content.typ": canvit-overview-diagram

#canvit-overview-diagram(
  glimpse-width: 1.42cm,
  shown: int(sys.inputs.at("shown")),
  panel-labels: false,
  first-index: 1,
  time-label: none,
)
