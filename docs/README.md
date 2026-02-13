# collAIder Website

A modern, responsive website for the collAIder project built with HTML and CSS.

## Features

- **Dark slate/charcoal theme** with purple accents
- **Multi-page structure** with clean navigation
- **Fully responsive design** for mobile, tablet, and desktop
- **Simple, elegant interactions** with purple highlights
- **Clean, professional layout** optimized for GitHub Pages

## Pages

1. **Home** (`index.html`) - Project overview with about section
2. **Demo** (`demo.html`) - Interactive demonstrations and use cases
3. **Team** (`team.html`) - Core research team with photo placeholders
4. **Funding** (`funding.html`) - Current grants and funding organizations
5. **Publications** (`publications.html`) - Featured research papers

## Setup Instructions for GitHub Pages

### Option 1: Enable GitHub Pages from your repository (Recommended)

1. **Upload your files to the repository:**
   - Place `collAIder_logo.png` in the root directory
   - Upload all HTML files (`index.html`, `demo.html`, `team.html`, `funding.html`, `publications.html`)
   - Upload `styles.css`
   - **Create a `photos` folder and add team member photos:**
     - Name them `member1.jpg`, `member2.jpg`, `member3.jpg`, etc.
     - Recommended: Square images, at least 300x300 pixels
     - Supported formats: .jpg, .png, .webp

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
├── photos/              # Team member photos
│   ├── member1.jpg
│   ├── member2.jpg
│   └── member3.jpg
└── README.md            # This file
```

## Adding Team Photos

To add team member photos:

1. Create a `photos` folder in your repository
2. Add your team member photos with these names:
   - `member1.jpg` - First team member
   - `member2.jpg` - Second team member
   - `member3.jpg` - Third team member
   - Add more as needed (member4.jpg, member5.jpg, etc.)
3. **Photo requirements:**
   - Square aspect ratio recommended (e.g., 300x300px or larger)
   - Supported formats: .jpg, .png, .webp
   - File size: Keep under 500KB for fast loading
4. Update the `team.html` file if you need to add more team members by copying the card structure

## Customization

### Updating Content

Each HTML page has placeholder content:
- **Team page**: Replace photo paths and update names, titles, and bios
- **Funding page**: Add your grant information and funding organization details
- **Publications page**: Add your research papers with links
- **Demo page**: Add your demo content, videos, or interactive elements
- **Home page**: Update the hero section and about content to match your project

### Changing Colors

All colors are defined in CSS variables at the top of `styles.css`:

```css
:root {
    --bg-primary: #1a1d2e;        /* Main background */
    --bg-secondary: #252836;      /* Card backgrounds */
    --bg-tertiary: #2d3142;       /* Alternate sections */
    --text-primary: #ffffff;      /* Main text */
    --text-secondary: #b8c1ec;    /* Secondary text */
    --accent: #a855f7;            /* Accent color (purple) */
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
