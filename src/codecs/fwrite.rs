use std::io::Write;

/// Trait to represent a writer which can be finalized:
/// for example, if something like a checksum must be written at the end of a stream.
pub trait FinalWrite: Write {
    /// Perform anything you might need to do at the end of a stream.
    ///
    /// If any additional bytes were successfully written, returns how many.
    /// By default, attempts no extra writes and finishes successfully.
    ///
    /// Is not guaranteed to only be called once.
    fn finalize(&mut self) -> std::io::Result<usize>;
}

pub struct FinalWriter<W: Write> {
    w: W,
}

impl<W: Write> FinalWriter<W> {
    pub fn new(w: W) -> Self {
        Self { w }
    }
}

impl<W: Write> Write for FinalWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.w.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.w.flush()
    }
}

impl<W: Write> FinalWrite for FinalWriter<W> {
    fn finalize(&mut self) -> std::io::Result<usize> {
        Ok(0)
    }
}
