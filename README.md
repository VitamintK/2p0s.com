# 2p0s.com

A research reference for two-player zero-sum games.

## Development

```bash
npm install
npm start        # dev server at http://localhost:8080 with live reload
npm run build    # build to _site/
```

You can also open `_site/index.html` directly in a browser without a server.

## Adding a game

Create a new markdown file in `src/games/`:

```markdown
---
layout: base.njk
tags: game
name: Kuhn Poker
info_states: 12
num_actions: 2
---

Game content here.
```

It will appear automatically in the index table.

## Deployment

Pushes to `main` deploy automatically to GitHub Pages via GitHub Actions.
Enable Pages in repo Settings → Pages → Source: **GitHub Actions**.
