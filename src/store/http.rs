use std::io::{ErrorKind, Read};
use std::iter::repeat_with;
use std::{collections::HashMap, io};

use bytes::{Buf, Bytes};
use httparse::parse_headers;
use itertools::Itertools;
use reqwest::blocking::Response;
use reqwest::Method;
use reqwest::{
    blocking::{Client, RequestBuilder},
    IntoUrl, StatusCode, Url,
};

use super::{NodeKey, ReadableStore, Store};

pub struct HttpStore {
    client: Client,
    base_url: Url,
    basic_auth: Option<(String, String)>,
}

impl HttpStore {
    /// `base_url` must end with a `/`.
    /// `client` should be constructed with any required headers
    /// (see [reqwest::blocking::ClientBuilder]).
    pub fn new<U: IntoUrl>(
        client: Client,
        base_url: U,
        basic_auth: Option<(String, String)>,
    ) -> reqwest::Result<Self> {
        Ok(Self {
            client,
            base_url: base_url.into_url()?,
            basic_auth,
        })
    }

    fn make_request_builder(
        &self,
        method: Method,
        key: &NodeKey,
    ) -> Result<RequestBuilder, String> {
        let encoded = key.encode();
        let url = self.base_url.join(&encoded).map_err(|e| e.to_string())?;
        let mut builder = self.client.request(method, url);
        if let Some((u, p)) = &self.basic_auth {
            builder = builder.basic_auth(u, Some(p));
        }
        Ok(builder)
    }
}

impl Store for HttpStore {}

impl ReadableStore for HttpStore {
    type Readable = reqwest::blocking::Response;

    fn get(&self, key: &NodeKey) -> io::Result<Option<Self::Readable>> {
        let builder = self
            .make_request_builder(Method::GET, key)
            .map_err(|_e| io::Error::new(ErrorKind::InvalidInput, "Could not create URL"))?;

        match builder.send() {
            Ok(r) => Ok(Some(r)),
            Err(e) => {
                if e.status().unwrap() == StatusCode::NOT_FOUND {
                    Ok(None)
                } else {
                    Err(io::Error::new(ErrorKind::Other, e))
                }
            }
        }
    }

    fn get_partial_values(
        &self,
        key_ranges: &[(NodeKey, crate::RangeRequest)],
    ) -> io::Result<Vec<Option<Box<dyn io::Read>>>> {
        let mut ranges = HashMap::with_capacity(key_ranges.len());

        // group ranges by the node key, storing the original index
        for (idx, (k, r)) in key_ranges.iter().enumerate() {
            ranges.entry(k).or_insert(Vec::default()).push((idx, r));
        }

        // pre-allocate output vec with Nones (can't use vec![] as sizes of Some variants are unknown)
        let mut out: Vec<Option<Box<dyn Read>>> =
            repeat_with(|| None).take(key_ranges.len()).collect();

        for (k, idx_rs) in ranges.into_iter() {
            // make a single request per node
            let req = "bytes=".to_string() + &idx_rs.iter().map(|(_, r)| r.to_string()).join(", ");
            let builder = self
                .make_request_builder(Method::GET, k)
                .map_err(|_e| io::Error::new(ErrorKind::InvalidInput, "Could not create URL"))?
                .header("range", &req);

            if let Some(r) = map_response_err(builder.send())? {
                let status = r.status();

                // need content type first as .bytes consumes header
                let content_type = r
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .map(|h| h.clone());

                // read eagerly so that we can easily slice later
                let bytes = r.bytes().map_err(|e| io::Error::new(ErrorKind::Other, e))?;

                if status == StatusCode::PARTIAL_CONTENT {
                    // server knows how to deal with partial content

                    if idx_rs.len() == 1 {
                        // single ranges look different
                        let idx = idx_rs[0].0;
                        out[idx] = Some(Box::new(bytes.reader()) as Box<dyn Read>);
                    } else if let Some(ct) = content_type {
                        // actual multi ranges
                        let bound =
                            get_boundary(ct.to_str().expect("content type was invalid str"))
                                .expect("Invalid multipart content header");
                        let parts = split_multipart_bytes(bytes, bound)
                            .map_err(|e| io::Error::new(ErrorKind::Other, e))?;

                        for (p, (idx, _)) in parts.into_iter().zip(idx_rs.iter()) {
                            out[*idx] = Some(Box::new(p.reader()) as Box<dyn Read>);
                        }
                    } else {
                        return Err(io::Error::new(
                            ErrorKind::Other,
                            "No multipart content header",
                        ));
                    }
                } else if status == StatusCode::OK {
                    // server doesn't know how to do partial responses, returned whole chunk
                    for (idx, rreq) in idx_rs.iter() {
                        let rdr = bytes.slice(rreq.to_range(bytes.len())).reader();
                        out[*idx] = Some(Box::new(rdr) as Box<dyn Read>);
                    }
                } else {
                    return Err(io::Error::new(ErrorKind::Other, "Unknown status code"));
                }
            } else {
                // 404, empty chunk
                continue;
            }
        }

        Ok(out)
    }
}

fn get_boundary<'a>(content_type: &'a str) -> Option<&'a str> {
    let mut cti = content_type.split(';').map(|s| s.trim());
    let t = cti.next()?;
    if t != "multipart/byteranges" {
        return None;
    }
    let bound = cti.next()?;
    let mut bi = bound.split('=').map(|s| s.trim());
    if bi.next()? != "boundary" {
        return None;
    }
    bi.next()
}

fn map_response_err(response: reqwest::Result<Response>) -> io::Result<Option<Response>> {
    match response {
        Ok(r) => Ok(Some(r)),
        Err(e) => {
            if e.status().unwrap() == StatusCode::NOT_FOUND {
                Ok(None)
            } else {
                Err(io::Error::new(ErrorKind::Other, e))
            }
        }
    }
}

enum ContentRangeError {
    InvalidPrefix(),
}

// or just use regex?
fn parse_content_range(range_value: &str) -> Option<(usize, usize)> {
    let mut parts = range_value.split_ascii_whitespace();
    let prefix = parts.next().unwrap();
    if prefix != "bytes" {
        panic!("Content range is not in bytes");
    }
    let range = parts.next().unwrap();
    let mut range_parts = range.split('/');
    let start_stop = range_parts.next().unwrap();
    if start_stop == "*" {
        return None;
    }
    let mut ss_it = start_stop.split('-');
    let start_str = ss_it.next().unwrap();
    let start: usize = start_str.parse().unwrap();
    let stop_str = ss_it.next().unwrap();
    let stop = stop_str.parse::<usize>().unwrap() + 1;

    Some((start, stop))
}

fn remove_header(content: Bytes) -> Result<Bytes, &'static str> {
    let mut buf = [httparse::EMPTY_HEADER; 50];

    let offset =
        match parse_headers(&content[..], &mut buf).map_err(|_| "could not parse header")? {
            httparse::Status::Complete((idx, _)) => idx,
            httparse::Status::Partial => return Err("only partially successful"),
        };
    Ok(content.slice(offset..))
}

fn split_multipart_bytes(content: Bytes, boundary: &str) -> Result<Vec<Bytes>, &'static str> {
    let mut out = Vec::default();
    let bound_string = format!("\r\n\r\n--{boundary}\r\n");
    let bound = bound_string.as_bytes();
    let mut start = None;

    for (end, w) in content[..].windows(boundary.len()).enumerate() {
        if w != bound {
            continue;
        }

        if let Some(s) = start {
            out.push(remove_header(content.slice(s..end))?);
        }
        start = Some(end + bound.len());
    }

    if let Some(s) = start {
        out.push(remove_header(content.slice(s..))?);
    }

    Ok(out)
}
