# PaperLens Design System Specification (DESIGN.md)

## 1. Visual Identity & Product Character

PaperLens is an editorial research intelligence workspace for scholars, scientists, and machine learning researchers analyzing and synthesizing academic literature.

### Brand Character
- **Calm, authoritative, and academic:** Reads like a peer-reviewed research monograph or serious technical publishing house (*Nature*, *MIT Press*, *Overleaf*).
- **Clear typographic hierarchy:** Emphasizes high-contrast legibility for long-form academic text.
- **Instrument-grade clarity:** Every surface, divider, and token serves comprehension. No decorative fluff, no neon glows, no synthetic AI tropes.

---

## 2. Typography Strategy

PaperLens employs three typographic voices with strict functional separation:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Editorial Serif (Newsreader / Source Serif 4)                 │
│    Used for: Paper titles, research digests, abstractive summaries│
├─────────────────────────────────────────────────────────────────┤
│ 2. Interface Sans (Geist / Inter Tight / System Sans)           │
│    Used for: Navigation, controls, buttons, labels, tabs, modals │
├─────────────────────────────────────────────────────────────────┤
│ 3. Tabular Monospace (JetBrains Mono / Geist Mono)              │
│    Used for: Token counts, ROUGE scores, runtimes, hashes, IDs   │
└─────────────────────────────────────────────────────────────────┘
```

### Scale & Hierarchy

| Role | Font Family | Size | Weight | Tracking | Leading | Max Measure |
|---|---|---|---|---|---|---|
| **Display Title** | Serif | `2.0rem – 2.25rem` (32–36px) | 600 (SemiBold) | `-0.02em` | `1.2` | 45ch |
| **Section Heading** | Sans | `1.125rem` (18px) | 600 (SemiBold) | `-0.01em` | `1.3` | 60ch |
| **Digest Body** | Serif | `1.0rem` (16px) | 400 (Regular) | `0em` | `1.75` | 68ch |
| **UI Control / Label** | Sans | `0.875rem` (14px) | 500 (Medium) | `-0.01em` | `1.4` | — |
| **Micro Caption / Tag** | Sans | `0.75rem` (12px) | 500 (Medium) | `+0.02em` | `1.3` | — |
| **Metric / Code** | Mono | `0.8125rem` (13px) | 400 / 500 | `0em` | `1.4` | Tabular numbers |

---

## 3. Color & Surface Architecture

The palette uses a warm editorial paper scale in light mode and a deep obsidian slate scale in dark mode. Saturation is kept strictly under 75% for all accents.

### CSS Custom Properties & Semantic Tokens

```css
:root {
  /* Canvas & Surfaces */
  --bg-canvas: #fbfbf9;            /* Warm parchment background */
  --bg-surface: #ffffff;           /* Clean panel surface */
  --bg-surface-subtle: #f3f3ee;    /* Inset/secondary background */
  --bg-surface-elevated: #ffffff;  /* Modal / popover surface */

  /* Borders & Rules */
  --border-subtle: #e5e5de;        /* Hairline structural dividers */
  --border-strong: #cacac0;        /* Active / focused control borders */

  /* Typography */
  --text-primary: #121316;         /* High-contrast ink black */
  --text-secondary: #4b5262;       /* Mid-tone slate for supporting copy */
  --text-muted: #787f91;           /* Quiet caption & label text */

  /* Accents */
  --accent-primary: #1e3a8a;       /* Deep Oxford academic blue */
  --accent-surface: #eff4fe;       /* Subtle blue tint for active tabs/chips */
  --accent-hover: #172554;         /* Hover state on primary button */

  /* Feedback */
  --status-success: #065f46;       /* Editorial forest green */
  --status-success-bg: #ecfdf5;
  --status-warning: #92400e;       /* Warm amber */
  --status-warning-bg: #fffbeb;
  --status-error: #991b1b;         /* Deep crimson */
  --status-error-bg: #fef2f2;
}

.dark {
  /* Canvas & Surfaces */
  --bg-canvas: #0c0e12;            /* Deep obsidian dark */
  --bg-surface: #14171d;           /* Dark panel surface */
  --bg-surface-subtle: #1a1e26;    /* Inset/secondary background */
  --bg-surface-elevated: #20242e;  /* Modal / popover surface */

  /* Borders & Rules */
  --border-subtle: #242934;        /* Hairline structural dividers */
  --border-strong: #373e4d;        /* Active / focused control borders */

  /* Typography */
  --text-primary: #f1f3f7;         /* Soft white reading text */
  --text-secondary: #949daf;       /* Mid-tone slate */
  --text-muted: #646e82;           /* Quiet caption & label text */

  /* Accents */
  --accent-primary: #3b82f6;       /* Bright slate blue */
  --accent-surface: #172554;       /* Dark slate blue tint */
  --accent-hover: #60a5fa;

  /* Feedback */
  --status-success: #34d399;
  --status-success-bg: #064e3b;
  --status-warning: #fbbf24;
  --status-warning-bg: #78350f;
  --status-error: #f87171;
  --status-error-bg: #7f1d1d;
}
```

---

## 4. Spacing, Geometry & Depth

- **Layout Grid:** Strict 8px base rhythm (`gap-2` = 8px, `gap-4` = 16px, `gap-6` = 24px, `gap-8` = 32px).
- **Max Width Container:** Application surfaces are constrained to `max-w-7xl mx-auto px-6 lg:px-8`.
- **Corner Radii:**
  - Standard cards and panels: `rounded-lg` (8px)
  - Controls, inputs, and buttons: `rounded-md` (6px)
  - Badges, pills, tags: `rounded-full`
  - *No oversized 24px/32px cartoon radii.*
- **Borders & Dividers:** Hairline `1px` borders only. Shadows are restrained and tinted:
  - Default elevation: `shadow-sm`
  - Popover / Modal elevation: `shadow-md shadow-black/5 dark:shadow-black/30`
  - *No glow effects, neon halos, or heavy diffuse drop-shadows.*

---

## 5. Component States & Feedback

Every interactive element implements explicit lifecycle states:

1. **Default:** Crisp border, neutral text, subtle background.
2. **Hover:** 1px border darkening, subtle background tint (`hover:bg-slate-50 dark:hover:bg-slate-800/40`), cursor pointer.
3. **Active / Pressed:** Physical mechanical feedback: `active:scale-[0.98]` or `active:translate-y-[0.5px]`.
4. **Focused:** High-visibility focus ring for keyboard accessibility:
   `focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-600 focus-visible:ring-offset-2`.
5. **Disabled:** Opacity 50%, `cursor-not-allowed`, interactions neutralized.
6. **Loading:**
   - Honest indeterminate loading spinners or shimmering skeletal bars matching the content shape.
   - **No fake percentage counters or simulated progress increments.**
7. **Empty:** Composed, informative empty states with clear icon and primary CTA (e.g. *"No papers ingested yet. Add your first PDF to begin."*).

---

## 6. Motion & Micro-Interactions

Motion must communicate hierarchy and state changes rather than act as decoration:

- **Duration:** 150ms – 250ms for micro-interactions (buttons, dropdowns, tabs). 300ms for panel transitions.
- **Timing Function:** `cubic-bezier(0.16, 1, 0.3, 1)` (ease-out with natural dampening).
- **Properties Animated:** Exclusively `transform` and `opacity`. Never `width`, `height`, `margin`, or `left`.
- **Reduced Motion:** Fully obeys `@media (prefers-reduced-motion: reduce)`:
  ```css
  @media (prefers-reduced-motion: reduce) {
    *, ::before, ::after {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
      scroll-behavior: auto !important;
    }
  }
  ```

---

## 7. Responsive Behavior

- **Desktop (≥ 1024px):** Full dual-pane reading workspace with sticky navigation and 70/30 main-to-rail layout.
- **Tablet (768px – 1023px):** Single-column stack with tabs or drawers for the contextual inspector rail.
- **Mobile (< 768px):** Clean header with collapsible drawer menu. Horizontal tables (matrix comparison) switch to stacked dimension cards.

---

## 8. Accessibility Guidelines (WCAG AA Compliance)

1. **Color Contrast:** All body text meets minimum 4.5:1 contrast against its background. Large headings meet 3:1.
2. **Form Labels:** Every input has a visible, associated `<label>` element. Placeholders are never used as labels.
3. **Screen Readers:** Explicit ARIA roles on tabs (`role="tab"`, `aria-selected`, `aria-controls`), accordions (`aria-expanded`), and modals.
4. **Single-line CTAs:** Primary button text never wraps at desktop viewports.
5. **No Color Alone:** Status indicators combine color with a clear icon and text label.

---

## 9. Explicit DO / DON'T Rules

### DO:
- **DO** present the 6 structured digest sections (Problem, Methodology, Dataset, Results, Limitations, Conclusion) with editorial clarity.
- **DO** mark missing sections as `[Not reported in document]` rather than hiding them or fabricating content.
- **DO** display token counts, runtimes, and ROUGE metrics in tabular monospace numbers (`font-mono tabular-nums`).
- **DO** use honest, indeterminate loading indicators during model inference.
- **DO** maintain strict visual consistency across light and dark modes.

### DON'T:
- **DON'T** use purple/pink "AI gradient" glows, blurred mesh backgrounds, or glassmorphic cards.
- **DON'T** implement fake percentage counters (e.g. 25% → 50% → 75%) that simulate progress the backend does not report.
- **DON'T** display fake team members, mock user profiles, or fictional persona avatars.
- **DON'T** put permanent hardware badges (like "RTX 4050") in the primary navigation.
- **DON'T** wrap button labels to multiple lines.
- **DON'T** add arbitrary version chips (e.g. "v2.0") to the brandmark.
- **DON'T** introduce heavy JS frameworks (React, Vue, Next) or build tools. Keep the single HTML file lightweight and robust.
