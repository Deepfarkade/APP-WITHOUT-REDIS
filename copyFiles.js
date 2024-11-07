import fs from 'fs-extra';

async function copyFiles() {
    try {
        // Copy the entire dist directory to backend/static
        await fs.copy('dist', 'backend/static');
        
        // Ensure index.html is in the root of static
        await fs.copy('dist/index.html', 'backend/static/index.html');
        
        console.log('Files copied successfully!');
    } catch (err) {
        console.error('Error copying files:', err);
        process.exit(1);
    }
}

copyFiles();