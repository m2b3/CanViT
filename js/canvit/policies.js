// The paper's viewing policies, keyed by the ids its exported data uses, with the paper's labels and figure colors.
export const POLICIES = {
  coarse_to_fine: { label: "C2F", name: "coarse-to-fine", color: "#1f77b4" },
  fine_to_coarse: { label: "F2C", name: "fine-to-coarse", color: "#d62728" },
  entropy_coarse_to_fine: { label: "EG-C2F", name: "entropy-guided coarse-to-fine", color: "#2ca02c" },
  random: { label: "R-IID", name: "random", color: "#ff7f0e" },
  full_then_random: { label: "F-IID", name: "full, then random", color: "#1a1a1a" },
  repeated_full_scene: { label: "RFS", name: "repeated full scene", color: "#8c564b" },
};
