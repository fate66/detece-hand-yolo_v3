const Koa = require("koa");
const app = new Koa();
const Router = require("koa-router");
const index = require("./router/index");
const { koaBody } = require("koa-body");
// app.use(async (ctx) => {
//   ctx.body = "Hello World";
// });

const cors = require("koa-cors");
app.use(
  cors({
    origin: "*",
  })
);

app.use(
  koaBody({
    multipart: true,
  })
);
global.start = { top: 0, left: 0 };

const router = new Router();
router.use(index.routes());

// 启动路由
app.use(router.routes()).use(router.allowedMethods());

app.listen(3000);
