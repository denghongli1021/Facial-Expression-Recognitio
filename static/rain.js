const rainContainer = document.getElementById('rain-container');
let rainInterval;

// 動態生成雨滴
function createRain() {
    const numberOfRaindrops = 50; // 每次生成的雨滴數量
    for (let i = 0; i < numberOfRaindrops; i++) {
        const raindrop = document.createElement('div');
        raindrop.classList.add('raindrop');
        raindrop.style.left = Math.random() * 100 + 'vw'; // 隨機水平位置
        raindrop.style.animationDuration = (Math.random() * 0.5 + 1) + 's'; // 隨機掉落速度
        rainContainer.appendChild(raindrop);

        // 當雨滴動畫結束時，自動移除雨滴
        raindrop.addEventListener('animationend', () => {
            raindrop.remove();
        });
    }
}

// 啟用下雨效果
function startRain() {
    stopRain(); // 確保不會重複啟用
    rainInterval = setInterval(createRain, 300);
}

// 停止下雨效果
function stopRain() {
    clearInterval(rainInterval);
    rainContainer.innerHTML = ''; // 清空雨滴
}
