/** "ff8000..." -> Uint8Array [255, 128, 0, ...] */
export const hexBytes = (hex) => Uint8Array.from(hex.match(/../g), (h) => parseInt(h, 16));
