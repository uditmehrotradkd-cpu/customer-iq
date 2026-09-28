// Stock media is self-hosted (see /static/media). All items are from Pexels under the Pexels License:
// free for commercial use, no attribution required; credited here as good practice.
const MEDIA = "/static/media";

export const HERO_VIDEO = { src: `${MEDIA}/hero-data.mp4`, poster: `${MEDIA}/hero-data-poster.jpg` };
export const BAGS_VIDEO = { src: `${MEDIA}/agent-particles.mp4`, poster: `${MEDIA}/agent-particles-poster.jpg` };
export const EXPLORER_IMAGE = `${MEDIA}/explorer-data.jpg`;

const PERSONAS = {
  "Premium Campaign Responders": { image: `${MEDIA}/persona-premium.jpg`, alt: "Neon-lit fashion store window", tagline: "Big baskets, loves a great campaign" },
  "Affluent Quiet Loyalists": { image: `${MEDIA}/persona-loyalist.jpg`, alt: "Close-up of gold and premium credit cards", tagline: "Steady, high-value, ignores the noise" },
  "Digital Deal-Seekers": { image: `${MEDIA}/persona-digital.jpg`, alt: "Online shop on a laptop with a phone ready to pay", tagline: "Web-first, browsing for the best deal" },
  "Budget-Conscious Browsers": { image: `${MEDIA}/persona-budget.jpg`, alt: "Phone calculator beside a pile of coins", tagline: "Everyday value, frequent visits" },
};
const FALLBACK = { image: `${MEDIA}/pool-workspace.jpg`, alt: "Trading workspace with live charts on screens", tagline: "A distinct behavioural segment" };
// Segments of an uploaded dataset have no fixed persona, so each one gets a distinct picture by position.
const POOL = [
  PERSONAS["Premium Campaign Responders"],
  PERSONAS["Affluent Quiet Loyalists"],
  PERSONAS["Digital Deal-Seekers"],
  PERSONAS["Budget-Conscious Browsers"],
  FALLBACK,
  { image: `${MEDIA}/pool-circuits.jpg`, alt: "Glowing blue circuit-like 3D shapes", tagline: "A distinct group in your data" },
  { image: `${MEDIA}/explorer-data.jpg`, alt: "Colourful data visualisation on a dark screen", tagline: "A distinct group in your data" },
  { image: `${MEDIA}/method-network.jpg`, alt: "Glowing blue fibre-optic light streams", tagline: "A distinct group in your data" },
];
export const METHOD_IMAGE = `${MEDIA}/method-network.jpg`;

export function persona(name, index = null) {
  if (PERSONAS[name]) return PERSONAS[name];
  if (index == null) return FALLBACK;
  const base = POOL[index % POOL.length];
  return { ...base, tagline: "A distinct group in your data" };
}

export const CREDITS = [
  { item: "Data dashboard video (hero)", author: "Pexels contributor", url: "https://www.pexels.com/video/futuristic-data-visualization-dashboard-34129037/" },
  { item: "Neural particles video (data agent)", author: "nicola-narracci", url: "https://www.pexels.com/video/stunning-neural-network-particle-animation-32446624/" },
  { item: "Neon store photo", author: "Pexels contributor", url: "https://www.pexels.com/photo/pink-neon-signage-1928079/" },
  { item: "Credit cards photo", author: "pixabay", url: "https://www.pexels.com/photo/close-up-photo-of-credit-cards-164501/" },
  { item: "E-commerce payment photo", author: "julio-lopez", url: "https://www.pexels.com/photo/modern-e-commerce-payment-interface-display-29502378/" },
  { item: "Calculator and coins photo", author: "polina-tankilevitch", url: "https://www.pexels.com/photo/a-mobile-phone-with-calculator-on-a-wooden-table-6927351/" },
  { item: "Data visualisation photo", author: "Pexels contributor", url: "https://www.pexels.com/photo/dynamic-financial-data-visualization-and-analysis-38808473/" },
  { item: "Blue neon fibres photo", author: "Pexels contributor", url: "https://www.pexels.com/photo/glowing-blue-neons-8640331/" },
  { item: "Blue circuit shapes photo", author: "steve", url: "https://www.pexels.com/photo/computer-graphics-in-color-blue-12537427/" },
  { item: "Trading workspace photo", author: "jakubzerdzicki", url: "https://www.pexels.com/photo/modern-digital-trading-workspace-with-charts-31738798/" },
];
