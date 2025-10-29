# Spacing Findings

## Computed spacing snapshot

| Element | Property | Computed value | Source |
| --- | --- | --- | --- |
| `.step` (all columns) | `margin-top` | `0px` | Streamlit base reset; no custom override |
| `.step` (all columns) | `margin-bottom` | `0.75rem` (`var(--fc-step-vspace)`) | `src/ui/theme/ui-components.css:3` |
| `[data-testid="stVerticalBlock"]` within columns | `row-gap` | `1rem` | Streamlit layout stylesheet (not overridden globally) |

> The values above reflect the defaults before any per-step overrides (e.g., the detect step tightening its inner `row-gap`).

## Conclusions

The large vertical spacing between steps primarily comes from two stacked contributors:

1. Streamlit sets a `row-gap` of `1rem` on every `[data-testid="stVerticalBlock"]`, separating the logical blocks that wrap each step inside a column.
2. Each `.step` adds its own `margin-bottom` of `0.75rem` via our shared `ui-components.css`. Together they yield roughly `1.75rem` (28px) of space between adjacent steps, which is what surfaces in the left column as well as the middle/right columns.

The new spacing probe (`?debug=spacing` or `SPACING_DEBUG=1`) can be used to confirm these metrics in the running app and to inspect the load order of the last five `<style>` tags to ensure our overrides appear after Streamlit's base CSS.
