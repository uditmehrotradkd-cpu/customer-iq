// Stock media is self-hosted (see /static/media). All items are from Pexels under the Pexels License:
// free for commercial use, no attribution required; credited here as good practice.
const MEDIA = "/static/media";

export const HERO_VIDEO = { src: `${MEDIA}/hero-mall.mp4`, poster: `${MEDIA}/hero-poster.jpg` };
export const BAGS_VIDEO = { src: `${MEDIA}/shopping-bags.mp4`, poster: `${MEDIA}/bags-poster.jpg` };
export const EXPLORER_IMAGE = `${MEDIA}/online-cart.jpg`;

const PERSONAS = {
  "Premium Campaign Responders": { image: `${MEDIA}/premium.jpg`, alt: "Shopper with several bags in a premium menswear store", tagline: "Big baskets, loves a great campaign" },
  "Affluent Quiet Loyalists": { image: `${MEDIA}/loyalist.jpg`, alt: "Curated wine shelves in a specialist shop", tagline: "Steady, high-value, ignores the noise" },
  "Digital Deal-Seekers": { image: `${MEDIA}/digital.jpg`, alt: "Online shopping on a laptop surrounded by shopping bags", tagline: "Web-first, browsing for the best deal" },
  "Budget-Conscious Browsers": { image: `${MEDIA}/budget.jpg`, alt: "Supermarket trolley in a grocery aisle", tagline: "Everyday value, frequent visits" },
};
const FALLBACK = { image: `${MEDIA}/online-cart.jpg`, alt: "Shopping trolley on a laptop", tagline: "A distinct behavioural segment" };
// Segments of an uploaded dataset have no fixed persona, so each one gets a distinct picture by position.
const POOL = [
  PERSONAS["Premium Campaign Responders"],
  PERSONAS["Affluent Quiet Loyalists"],
  PERSONAS["Digital Deal-Seekers"],
  PERSONAS["Budget-Conscious Browsers"],
  FALLBACK,
  { image: `${MEDIA}/bags-poster.jpg`, alt: "Shopping bags", tagline: "A distinct group in your data" },
  { image: `${MEDIA}/hero-poster.jpg`, alt: "People walking through a shopping mall", tagline: "A distinct group in your data" },
];
export const METHOD_IMAGE = `${MEDIA}/hero-poster.jpg`;

export function persona(name, index = null) {
  if (PERSONAS[name]) return PERSONAS[name];
  if (index == null) return FALLBACK;
  const base = POOL[index % POOL.length];
  return { ...base, tagline: "A distinct group in your data" };
}

export const CREDITS = [
  { item: "Mall video (hero)", author: "Pexels contributor", url: "https://www.pexels.com/video/people-spending-leisure-time-in-the-mall-4750090/" },
  { item: "Shopping bags video", author: "Pexels contributor", url: "https://www.pexels.com/download/video/7567908/" },
  { item: "Premium shopper photo", author: "gustavo-fring", url: "https://www.pexels.com/photo/man-in-brown-suit-sitting-on-ottoman-6050430/" },
  { item: "Wine shelves photo", author: "yagiz-ucal", url: "https://www.pexels.com/photo/wooden-wine-rack-with-diverse-bottles-in-cellar-32551644/" },
  { item: "Online shopping photo", author: "n-voitkevich", url: "https://www.pexels.com/photo/a-top-shot-of-a-woman-using-a-laptop-while-surrounded-by-paper-bags-6214129/" },
  { item: "Supermarket trolley photo", author: "eduschadesoares", url: "https://www.pexels.com/photo/shopping-cart-in-a-supermarket-5498233/" },
  { item: "Mini trolley on laptop photo", author: "karola-g", url: "https://www.pexels.com/photo/shopping-cart-on-a-macbook-5632382/" },
];
