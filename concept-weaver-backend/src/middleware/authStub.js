/**
 * authStub - lightweight auth placeholder
 * For MVP this middleware does not block requests.
 * To enforce auth: check the Authorization header (JWT / API key) and return 401 if invalid.
 *
 * Example:
 *   const token = req.header('Authorization');
 *   if (!token || token !== 'Bearer MYSECRET') return res.status(401).json({...});
 */
module.exports = function (req, res, next) {
  // Uncomment to enforce header-based stub auth for quick testing:
  // const header = req.header('authorization');
  // if (!header || header !== 'Bearer testtoken') {
  //   return res.status(401).json({ success: false, status: 'error', message: 'Unauthorized' });
  // }
  next();
};
