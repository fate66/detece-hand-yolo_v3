const Router = require("koa-router");
const router = new Router();
const path = require("path");
const fse = require("fs-extra");
const { PythonShell } = require("python-shell");

// router.get("/", async (ctx, next) => {
//   ctx.body = ctx.request;
// });

function runPY(dest) {
  let res, rej;
  let options = {
    args: [dest],
  };
  PythonShell.run("main.py", options, function (err, result) {
    if (err) {
      console.log("err", err);
      rej && rej(false);
    } else {
      console.log("finished", result);
      res && res(result);
    }
  });
  return new Promise((resolve, reject) => {
    res = resolve;
    rej = reject;
  });
}

router.post("/upload", async (ctx, next) => {
  const file = ctx.request.files.file;
  const dest = path.join(__dirname, "../upload", file.originalFilename); // 目标目录，没有没有这个文件夹会自动创建
  await fse.move(file.filepath, dest); // 移动文件
  const res = await runPY(dest);
  const imageReact = JSON.parse(res[2]);
  const result = {
    start: { top: global.start.top, left: global.start.left },
    end: { top: imageReact[1], left: imageReact[0] },
  };

  if (global.start.top === 0 && global.start.left === 0) {
    global.start = { left: imageReact[0], top: imageReact[1] };
  } else {
    result.distance = {
      leftMove: imageReact[0] - global.start.left,
      topMove: imageReact[1] - global.start.top,
    };
  }
  console.log(global.start);
  ctx.body = {
    result,
  };
});

module.exports = router;
