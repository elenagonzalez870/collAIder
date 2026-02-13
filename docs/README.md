# collAIder Website

A modern, responsive website for the collAIder project built with HTML and CSS.

## Features

- **Dark slate/charcoal theme** with electric cyan accents
- **Multi-page structure** with consistent navigation
- **Fully responsive design** for mobile, tablet, and desktop
- **Modern animations** and hover effects
- **Clean, professional layout** optimized for GitHub Pages

## Pages

1. **Home** (`index.html`) - Project overview and key features
2. **Demo** (`demo.html`) - Interactive demonstrations and use cases
3. **Team** (`team.html`) - Research team and collaborators
4. **Funding** (`funding.html`) - Grants and financial supporters
5. **Publications** (`publications.html`) - Research papers and preprints

## Setup Instructions for GitHub Pages

### Option 1: Enable GitHub Pages from your repository (Recommended)

1. **Upload your files to the repository:**
   - Place `collAIder_logo.png` in the root directory
   - Upload all HTML files (`index.html`, `demo.html`, `team.html`, `funding.html`, `publications.html`)
   - Upload `styles.css`

2. **Enable GitHub Pages:**
   - Go to your repository: https://github.com/elenagonzalez870/collAIder
   - Click on **Settings** (top navigation)
   - Scroll down to **Pages** in the left sidebar
   - Under **Source**, select **Deploy from a branch**
   - Choose **main** branch and **/ (root)** folder
   - Click **Save**

3. **Access your website:**
   - Your site will be published at: `https://elenagonzalez870.github.io/collAIder/`
   - It may take a few minutes for the site to build and deploy
   - You'll see a green success message with the URL once it's ready

### Option 2: Create a docs folder (Alternative)

If you want to keep your website files separate from your code:

1. Create a `docs` folder in your repository
2. Move all HTML, CSS, and image files into the `docs` folder
3. In GitHub Pages settings, select **main** branch and **/docs** folder
4. Save and wait for deployment

## File Structure

```
collAIder/
├── index.html           # Home page
├── demo.html            # Demo page
├── team.html            # Team page
├── funding.html         # Funding page
├── publications.html    # Publications page
├── styles.css           # Main stylesheet
├── collAIder_logo.png   # Project logo
└── README.md            # This file
```

## Customization

### Adding Your Logo

Replace `collAIder_logo.png` with your actual logo. For best results:
- Use PNG format with transparency
- Recommended height: 50-100px
- The logo will automatically adjust with the design

### Updating Content

Each HTML page has placeholder content marked with clear labels:
- Replace "Team Member Name" with actual names
- Add real photos by replacing the 👤 emoji placeholders
- Update grant information, publication details, etc.
- Modify the hero taglines to match your project

### Changing Colors

All colors are defined in CSS variables at the top of `styles.css`:

```css
:root {
    --bg-primary: #1a1d2e;        /* Main background */
    --bg-secondary: #252836;      /* Card backgrounds */
    --bg-tertiary: #2d3142;       /* Alternate sections */
    --text-primary: #ffffff;      /* Main text */
    --text-secondary: #b8c1ec;    /* Secondary text */
    --accent: #00d9ff;            /* Accent color (cyan) */
}
```

Simply modify these values to change the entire color scheme.

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## License

This website template is provided as-is for the collAIder project.

## Support

For issues or questions about the website:
- Open an issue on GitHub
- Contact the development team

---

**Note:** Remember to update the placeholder content with your actual project information before publishing!
