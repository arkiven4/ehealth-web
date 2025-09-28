const express = require("express");
const csrf = require("csurf");
const flash = require("connect-flash");
const path = require("path");
var bodyParser = require("body-parser");
var cors = require("cors");
const fs = require('fs');
const helmet = require('helmet');
const tbcareRoutes = require('./routes/tbcare');


const csrfMiddleware = (req, res, next) => {
  const shouldSkipCSRF = req.method === "POST" && skipCSRFPatterns.some(pattern => req.path.includes(pattern));
  if (shouldSkipCSRF) {
    return next();
  }
  return csrf()(req, res, (err) => {
    if (err) {
      console.error('CSRF Error:', err);
      return res.status(403).json({ error: 'Invalid CSRF token' });
    }
    next();
  });
};

var tbcough_folder = './public/uploads/batuk/'
if (!fs.existsSync(tbcough_folder)) {
  fs.mkdirSync(tbcough_folder, { recursive: true });
}

// Usefull Variable 
const routeNames = ['general', 'doctor', 'patient', 'admin', 'auth', 'api', 'form', 'vibio'];
const skipCSRFPatterns = [
  "/api/",
  "/form/",
  "/submit-data-batuk"
];

// import routing
const routes = {};
routeNames.forEach(name => {
  routes[name] = require(`./routes/${name}`);
});
const error_route = require(`./routes/error`);

// import helper
const rootdir = require("./helpers/path");
const initPraram = require("./helpers/init");

// import middleware
const parse = require("./middlewares/parsing");
const db = require("./middlewares/database");
const auth = require("./middlewares/auth");

const app = express();
const csrfProtection = csrf();

// set template engine
app.set("view engine", "ejs");
app.set("views", "views");
app.locals.moment = require("moment");

const corsOptions = {
  origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-CSRF-Token'],
  credentials: true
};
app.use(cors(corsOptions));

// parsing body/file and expose public dir
// app.use(parse.bodyJsonHandler);
// app.use(parse.bodyParserHandler);
// 

// app.use(
//   bodyParser.urlencoded({
//     extended: true,
//   })
// );
// app.use(bodyParser.json())
app.use(express.static(path.join(rootdir, "public")));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// API
app.use(routes.api);
app.use(routes.form);

// security & authentication
app.use(db.sessionMiddleware);
app.use(csrfMiddleware);
app.use(auth.clientAuth);

// notification
app.use(flash());


// TBCARE
app.use('/tbcare', tbcareRoutes); 

// routeNames.forEach(routeName => {
//   app.use(routes[routeName]);
// });

// routing request
routeNames.forEach(routeName => {
  app.use(routes[routeName]);
});

app.use((err, req, res, next) => {
  console.error(err.stack);
  if (err.code === 'EBADCSRFTOKEN') {
    return res.status(403).json({ error: 'Invalid CSRF token' });
  }
  res.status(500).json({ error: 'Something went wrong!' });
});
app.use(error_route);

app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "https:"]
    }
  }
}));

db.initMongoose(() => {
  app.listen(8080, () => {
    console.log(`Server running on port 8080`);
  });
});