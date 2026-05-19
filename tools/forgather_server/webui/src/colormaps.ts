/** Matplotlib-style sequential colormaps, evaluated from a polynomial
 *  fit so we don't ship a 256-entry LUT per map. The fits are the
 *  widely-cited ones from https://www.shadertoy.com/view/WlfXRN /
 *  "Polynomial approximations of perceptually uniform colormaps"
 *  (Inigo Quilez, samples by Matt Strine et al.). RMS error vs. the
 *  matplotlib source colormaps is well under one perceptual unit —
 *  indistinguishable to the eye at the swatch sizes we use.
 *
 *  Each colormap is a ``t in [0,1] → [r, g, b]`` function, each
 *  channel in [0,1]. ``cssColor`` clamps and formats as an HSL-free
 *  ``rgb(...)`` string suitable for ``style.background``.
 *
 *  Adding a new map: drop a new ``polyMap`` call with its six
 *  coefficient triplets into the registry below. No other code change
 *  needed. */

export type ColormapFn = (t: number) => [number, number, number];

export interface ColormapEntry {
  id: string;
  label: string;
  fn: ColormapFn;
}

function polyMap(
  c0: [number, number, number],
  c1: [number, number, number],
  c2: [number, number, number],
  c3: [number, number, number],
  c4: [number, number, number],
  c5: [number, number, number],
  c6: [number, number, number],
): ColormapFn {
  return (t: number): [number, number, number] => {
    const x = Math.min(1, Math.max(0, t));
    return [
      c0[0] + x * (c1[0] + x * (c2[0] + x * (c3[0] + x * (c4[0] + x * (c5[0] + x * c6[0]))))),
      c0[1] + x * (c1[1] + x * (c2[1] + x * (c3[1] + x * (c4[1] + x * (c5[1] + x * c6[1]))))),
      c0[2] + x * (c1[2] + x * (c2[2] + x * (c3[2] + x * (c4[2] + x * (c5[2] + x * c6[2]))))),
    ];
  };
}

const viridis = polyMap(
  [0.2777273272234177, 0.005407344544966578, 0.3340998053353061],
  [0.1050930431085774, 1.404613529898575, 1.384590162594685],
  [-0.3308618287255563, 0.214847559468213, 0.09509516302823659],
  [-4.634230498983486, -5.799100973351585, -19.33244095627987],
  [6.228269936347081, 14.17993336680509, 56.69055260068105],
  [4.776384997670288, -13.74514537774601, -65.35303263337234],
  [-5.435455855934631, 4.645852612178535, 26.3124352495832],
);

const plasma = polyMap(
  [0.05873234392399702, 0.02333670892565664, 0.5433401826748754],
  [2.176514634195958, 0.2383834171260182, 0.7539604599784036],
  [-2.689460476458034, -7.455851135738909, 3.110799939717086],
  [6.130348345893603, 42.3461881477227, -28.51885465332158],
  [-11.10743619062271, -82.66631109428045, 60.13984767418263],
  [10.02306557647065, 71.41361770095349, -54.07218655560067],
  [-3.658713842777788, -22.93153465461149, 18.19190778539828],
);

const inferno = polyMap(
  [0.0002189403691192265, 0.001651004631001012, -0.01948089843709184],
  [0.1065134194856116, 0.5639564367884091, 3.932712388889277],
  [11.60249308247187, -3.972853965665698, -15.9423941062914],
  [-41.70399613139459, 17.43639888205313, 44.35414519872813],
  [77.162935699427, -33.40235894210092, -81.80730925738993],
  [-71.31942824499214, 32.62606426397723, 73.20951985803202],
  [25.13112622477341, -12.24266895238567, -23.07032500287172],
);

const magma = polyMap(
  [-0.002136485053939582, -0.000749655052795221, -0.005386127855323933],
  [0.2516605407371642, 0.6775232436837668, 2.494026599312351],
  [8.353717279216625, -3.577719514958484, 0.3144679030132573],
  [-27.66873308576866, 14.26473078096533, -13.64921318813922],
  [52.17613981234068, -27.94360607168351, 12.94416944238394],
  [-50.76852536473588, 29.04658282127291, 4.23415299384055],
  [18.65570506591883, -11.48977351997711, -5.601961508734096],
);

const turbo = polyMap(
  [0.1140890109226559, 0.06288340699912215, 0.2248337216805064],
  [6.716419496985708, 3.182286745507602, 7.571581586103393],
  [-66.09402360453038, -4.9279827041226, -10.09439367561635],
  [228.7660791526501, 25.04986699771073, -91.54105330182436],
  [-334.8351565777451, -69.31749712757485, 288.5858850615712],
  [218.7637218434795, 67.52150567819112, -305.2045772184957],
  [-52.88903478218835, -21.54527364654712, 110.5174647748972],
);

/** Simple green→yellow→red HSL ramp — not perceptually uniform, but
 *  reads intuitively for "easy vs. hard" and works without computing
 *  RGB triplets. Kept as a fallback option in the menu. */
const greenRed: ColormapFn = (t: number) => {
  const x = Math.min(1, Math.max(0, t));
  // hue 120° (green) → 0° (red). Convert HSL(h, 60%, 32%) to RGB.
  return hslToRgb(120 * (1 - x), 0.6, 0.32);
};

export const COLORMAPS: ColormapEntry[] = [
  { id: "viridis", label: "viridis", fn: viridis },
  { id: "plasma", label: "plasma", fn: plasma },
  { id: "magma", label: "magma", fn: magma },
  { id: "inferno", label: "inferno", fn: inferno },
  { id: "turbo", label: "turbo", fn: turbo },
  { id: "green-red", label: "green→red", fn: greenRed },
];

export function getColormap(id: string): ColormapEntry {
  return COLORMAPS.find((c) => c.id === id) ?? COLORMAPS[0];
}

export function cssColor(rgb: [number, number, number]): string {
  const r = Math.round(255 * Math.min(1, Math.max(0, rgb[0])));
  const g = Math.round(255 * Math.min(1, Math.max(0, rgb[1])));
  const b = Math.round(255 * Math.min(1, Math.max(0, rgb[2])));
  return `rgb(${r}, ${g}, ${b})`;
}

/** Pick a readable foreground (black or white) for an arbitrary
 *  background RGB. Uses the standard relative-luminance formula —
 *  yellow/green ends of viridis/turbo are very bright and need black
 *  text; the dark purple/blue ends need white. */
export function readableForeground(rgb: [number, number, number]): string {
  const lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
  return lum > 0.55 ? "#111" : "#f0f0f0";
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  // Standard HSL → RGB; h in degrees.
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = (((h % 360) + 360) % 360) / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let r = 0,
    g = 0,
    b = 0;
  if (hp < 1) [r, g, b] = [c, x, 0];
  else if (hp < 2) [r, g, b] = [x, c, 0];
  else if (hp < 3) [r, g, b] = [0, c, x];
  else if (hp < 4) [r, g, b] = [0, x, c];
  else if (hp < 5) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  const m = l - c / 2;
  return [r + m, g + m, b + m];
}
