from flask import Flask, make_response, redirect, render_template, request, url_for, flash, g

app = Flask(__name__)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)