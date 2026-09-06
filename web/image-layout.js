// The grids draw a frame a few hundred device pixels wide. Asking for a downscaled copy keeps a
// board refresh from pulling a hundred multi-megabyte PNGs, which is what starved the
// transcript's own image of a connection right after a generation. The size snaps to the same
// ladder the server caches by (THUMBNAIL_SIZES in offcut_server.py — keep the two in step), so
// a dragged inspector only refetches when a rung is crossed rather than on every pixel. Ask in
// device pixels: on a 2x display a tile drawn at 309 CSS px needs 618 of them.
const THUMBNAIL_LADDER = [96, 128, 192, 256, 384, 512, 640, 768, 1024];

export function thumbnailUrl(image, cssSize) {
  const url = image?.image_url || image?.url || "";
  if (!url.startsWith("/api/image-files/")) return url;
  const wanted = Math.ceil(cssSize * (window.devicePixelRatio || 1));
  const step = THUMBNAIL_LADDER.find((size) => size >= wanted);
  return step ? `${url}?w=${step}` : url;
}

// The inspector is resizable and every grid reflows, so a tile's width is read off the laid-out
// grid rather than assumed. Before first layout the fallback stands in, and the next render
// corrects it.
export function gridColumnWidth(grid, fallback) {
  if (!grid) return fallback;
  // A laid-out grid resolves to a list of used widths ("310.5px 310.5px"). One that has never
  // been laid out — a panel still hidden — hands back the authored value instead, and reading a
  // number out of "repeat(auto-fill, minmax(74px, 1fr))" yielded the 1 of "1fr" and asked the
  // server for a one-pixel thumbnail. Only px tokens are measurements.
  const columns = getComputedStyle(grid).gridTemplateColumns.split(" ");
  const widths = columns.filter((column) => column.endsWith("px")).map((column) => parseFloat(column));
  const widest = Math.max(0, ...widths.filter(Number.isFinite));
  return widest > 0 ? widest : fallback;
}

// The rail shows whole frames instead of square crops. Rows are packed so every tile keeps its
// recorded aspect ratio and each row fills the rail's width exactly: there is no dead space to
// backfill, and visual order stays list order, which is what drag-to-reorder edits. Every size
// comes off the record, so the math never waits on a decoded bitmap.
// The gallery packs the same way against a page-wide column rather than a rail, which wants its
// own numbers: rows a fraction of that width rather than most of it, and a floor on tile width
// that is a flat readable size instead of a share of a much larger available.
const RAIL_PACK = {
  gap: 6, // mirrors the gap on .thumbnail-grid
  target: 0.6, // aim for rows about 60% of the rail wide — enough of a frame to read at a glance
  targetMin: 150,
  targetMax: 320,
  accept: 0.8, // a row may settle this far under the target rather than take one more frame
  loneCap: 0.85, // a row that cannot take a second frame stops this tall instead of filling
  loneCapMax: 540,
  // Narrow frames are judged by their width, not the row's height: three 9:16s can stand at a
  // healthy height and still be slivers, which is what "portrait images take up no space" was.
  // The floor walks between that and its own failure mode — set it high and a portrait can no
  // longer pair with a landscape, whose pair width runs near a quarter of the rail.
  minTile: (available) => 45 + available * 0.2,
};

export const GALLERY_PACK = {
  gap: 8, // mirrors the gap on .gallery-grid
  target: 0.2,
  targetMin: 190,
  targetMax: 340,
  accept: 0.78,
  loneCap: 0.4,
  loneCapMax: 420,
  minTile: (available) => Math.max(120, available * 0.06),
};

function railAspect(image) {
  const width = Number(image.width);
  const height = Number(image.height);
  return width > 0 && height > 0 ? width / height : 1;
}

export function packContentWidth(grid, fallback) {
  // A hidden panel reports 0; the fallback stands in and the ResizeObserver corrects on reveal.
  if (!grid || !grid.clientWidth) return fallback;
  const styles = getComputedStyle(grid);
  // clientWidth rounds, and a row that misses the content box by a fraction wraps its last tile
  // onto a line of its own, so give back a pixel and floor every width besides.
  return grid.clientWidth - parseFloat(styles.paddingLeft) - parseFloat(styles.paddingRight) - 1;
}

// Greedy justified packing, the Flickr/Google Photos shape: keep filling a row while it would
// still stand tall enough — and its narrowest frame still draw wide enough — at full rail width,
// then scale it to fit exactly. A row that could never take a second frame, a lone portrait or
// the ragged tail of the board, is capped rather than stretched to the whole width.
export function justifiedRailSizes(images, available, pack = RAIL_PACK) {
  const target = Math.min(pack.targetMax, Math.max(pack.targetMin, available * pack.target));
  const cap = Math.min(pack.loneCapMax, available * pack.loneCap);
  const minTile = pack.minTile(available);
  const rows = [];
  let row = [];
  let aspects = 0;
  let narrowest = Infinity;
  for (const image of images) {
    const aspect = railAspect(image);
    const count = row.length + 1;
    const fill = (available - pack.gap * (count - 1)) / (aspects + aspect);
    const narrowestAfter = Math.min(narrowest, aspect);
    // A row that cannot take this frame is finalized without it — unless that would strand a
    // lone tall frame below the cap beside a wide dead stripe. Pairing it with this frame, even
    // a touch narrower than the floor likes, always beats the stripe.
    const strands = row.length === 1 && available / aspects > cap;
    if (row.length && !strands && (fill < target * pack.accept || fill * narrowestAfter < minTile)) {
      rows.push({ row, aspects });
      row = [];
      aspects = 0;
      narrowest = Infinity;
    }
    row.push(image);
    aspects += aspect;
    narrowest = Math.min(narrowest, aspect);
  }
  if (row.length) rows.push({ row, aspects });
  const sizes = new Map();
  for (const { row, aspects } of rows) {
    const fill = (available - pack.gap * (row.length - 1)) / aspects;
    const height = Math.floor(Math.min(fill, cap));
    for (const image of row) sizes.set(image.id, { width: Math.floor(height * railAspect(image)), height });
  }
  return sizes;
}
