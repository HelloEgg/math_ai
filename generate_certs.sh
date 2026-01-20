#!/bin/bash

# Generate self-signed SSL certificates for development/testing
# For production, use Let's Encrypt instead!

CERT_DIR="certs"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"
DAYS_VALID=365

# Create certs directory if it doesn't exist
mkdir -p "$CERT_DIR"

# Check if certificates already exist
if [ -f "$CERT_FILE" ] && [ -f "$KEY_FILE" ]; then
    echo "Certificates already exist in $CERT_DIR/"
    echo "Delete them first if you want to regenerate:"
    echo "  rm -rf $CERT_DIR"
    exit 0
fi

echo "Generating self-signed SSL certificate..."
echo ""

# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -nodes \
    -keyout "$KEY_FILE" \
    -out "$CERT_FILE" \
    -days "$DAYS_VALID" \
    -subj "/C=US/ST=State/L=City/O=Organization/OU=Unit/CN=localhost"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ SSL certificates generated successfully!"
    echo ""
    echo "Files created:"
    echo "  Certificate: $CERT_FILE"
    echo "  Private key: $KEY_FILE"
    echo ""
    echo "To start the server with HTTPS:"
    echo "  python app.py --https"
    echo ""
    echo "Or with custom port:"
    echo "  python app.py --https --port 443"
    echo ""
    echo "WARNING: Self-signed certificates will show browser warnings."
    echo "For production, use Let's Encrypt:"
    echo "  sudo certbot certonly --standalone -d yourdomain.com"
else
    echo "Error generating certificates!"
    exit 1
fi
