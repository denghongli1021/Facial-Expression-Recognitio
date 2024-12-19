const sunContainer = document.getElementById('sun-container');

// 啟用笑臉太陽
function showSun() {
    if (!sunContainer) {
        console.error("Sun container not found!");
        return;
    }
    sunContainer.style.display = 'block';
    sunContainer.style.animation = 'scaleShine 2s infinite ease-in-out';
}

// 隱藏笑臉太陽
function hideSun() {
    if (!sunContainer) {
        console.error("Sun container not found!");
        return;
    }
    sunContainer.style.display = 'none';
    sunContainer.style.animation = 'none';
}
