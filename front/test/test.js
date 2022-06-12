import { Builder, By, Capabilities, until } from "selenium-webdriver";
describe("Wykrywa ruchy kamerką", () => {
  const driver = new Builder().withCapabilities(Capabilities.chrome()).build();
  it("Podłącza się do kamery", async () => {
    await driver.get("http://localhost:3000");
    let video = await driver.findElement(By.id("video"));
    let content = video.getAttribute("content"); // todo get content (real)
    setInterval(() => {
      let content_new = video.getAttribute("content");
      if (content_new !== content) {
        console.log("Camera detected move");
      }
    }, 1000 / 60);
  });
  after(async () => await driver.quit());
});
describe("Oblicza procent naładowania", () => {
  const driver = new Builder().withCapabilities(Capabilities.chrome()).build();
  it("Wciska przycisk i wyświetla procent naładowania", async () => {
    await driver.get("http://localhost:3000");
    let button = await driver.findElement(By.className("click-photo5"));
    await button.click();
    let x = 0;
    setInterval(() => {
      console.log(`${(x / 5000) * 100}%`);
      x += 1000 / 60;
      if (x >= 5000.0) {
        x = 0;
      }
    }, 1000 / 60);
  });
  after(async () => await driver.quit());
});
