exports.getLandingPage = (req, res, next) => {
  res.render('tbcare/landing', {
    pageTitle: 'Welcome to TBCare',
    isAuthenticated: req.session.isLoggedIn 
  });
};