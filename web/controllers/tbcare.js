exports.getLandingPage = (req, res, next) => {
  res.render('tbcare/landing', {
    pageTitle: 'Welcome to TBCare',
    isAuthenticated: req.session.isLoggedIn 
  });
};

exports.getDownloadPage = (req, res, next) => {
  res.render('tbcare/download', {
    pageTitle: 'Download TBCare App',
    isAuthenticated: req.session.isLoggedIn
  });
};

exports.getAboutPage = (req, res, next) => {
  res.render('tbcare/about', {
    pageTitle: 'About TBCare',
    isAuthenticated: req.session.isLoggedIn
  });
};

// exports.getTbcareLoginPage = (req, res, next) => {
//   res.render('tbcare/login', {
//     pageTitle: 'TBCare Login',
//     csrfToken: req.csrfToken() 
//   });
// };